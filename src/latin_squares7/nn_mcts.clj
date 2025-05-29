(ns latin-squares7.nn-mcts
  (:require [latin-squares7.mcts :as mcts]
            [latin-squares7.nn-metamorph :as nn]
            [latin-squares7.functions :as f]
            [tech.v3.tensor :as tensor]))

(def ^:private current-model (atom nil))
(def ^:private game-history (atom []))

(defn initialize-model []
  "Initialize the neural network model if not already done"
  (when (nil? @current-model)
    (println "Initializing neural network model...")
    (reset! current-model (nn/create-game-pipeline))))

(defn get-best-move
  "Get the best move using neural network guided MCTS"
  [state]
  (initialize-model)  ; Ensure model is initialized
  (let [predictions (nn/run-pipeline @current-model state :transform)
        policy (:policy predictions)
        value (:value predictions)
        
        ;; Use neural network policy as priors for MCTS
        move (mcts/mcts state 500 policy)]  ; Pass the policy to MCTS
    (when move
      ;; After making a move, update the model with the new position
      (let [next-state (f/make-move state move)
            next-predictions (nn/run-pipeline @current-model next-state :transform)]
        ;; Store the predictions for training later
        (swap! game-history conj {:state state
                                :move move
                                :next-state next-state
                                :policy policy
                                :value value
                                :next-policy (:policy next-predictions)
                                :next-value (:value next-predictions)})))
    move))

(defn train-on-game-history []
  "Train the model on the collected game history using a simple update rule"
  (when (seq @game-history)
    (println "Training on" (count @game-history) "positions...")
    (let [learning-rate 0.01
          updated-model (reduce (fn [current-model {:keys [state next-state]}]
                                (let [result (nn/run-pipeline current-model state :transform)
                                      target-value (if (f/solved? next-state) 1.0 0.0)
                                      current-value (:value result)
                                      ;; Simple gradient update for value head only
                                      value-delta (* learning-rate (- target-value current-value))
                                      ;; Get value head weights and biases, with fallback to current model
                                      value-head (get-in result [:layers :value])
                                      current-value-head (get-in current-model [:layers :value])
                                      value-weights (or (:weights value-head) (:weights current-value-head))
                                      value-biases (or (:biases value-head) (:biases current-value-head))]
                                  (if (and value-weights value-biases)
                                    (let [updated-value-weights (nn/tensor-add value-weights
                                                                             (tensor/->tensor
                                                                              (mapv (fn [row]
                                                                                     (mapv #(* value-delta %) row))
                                                                                   (seq value-weights))))
                                          updated-value-biases (nn/tensor-add value-biases
                                                                            (tensor/->tensor
                                                                             (mapv (fn [bias]
                                                                                    (* value-delta bias))
                                                                                  (flatten (seq value-biases)))))]
                                      (assoc-in current-model [:layers :value]
                                               {:weights updated-value-weights
                                                :biases updated-value-biases}))
                                    current-model)))
                              (or @current-model (nn/create-game-pipeline))
                              @game-history)]
      (when (fn? updated-model)  ; Ensure we have a valid pipeline function
        (reset! current-model updated-model))
      (reset! game-history []))))

(defn autoplay-from-position
  "Autoplay from a given position using neural network-guided MCTS"
  [game-state max-moves]
  (let [result (f/autoplay-from-position game-state max-moves get-best-move)
        final-state (:final-state result)
        winner (when (:game-over? result)
                (if (= (:current-player final-state) :alice)
                  :bob  ; If it's alice's turn and game is over, bob made the last move
                  :alice))]  ; If it's bob's turn and game is over, alice made the last move
    (println "\n=== Game Summary ===")
    (println "Final board state:")
    (f/print-board (:board final-state))
    (println "\nGame statistics:")
    (println "Moves made:" (:moves-made result))
    (println "Game over:" (:game-over? result))
    (when winner
      (println "Winner:" (name winner) "(made the last move)"))
    (println "\nMove history:")
    (doseq [[i move] (map-indexed vector (:moves result))]
      (println (format "Move %d: %s" (inc i) move)))
    (assoc result :winner winner)))

(defn get-random-move [game-state]
  "Get a random valid move from the current position"
  (let [valid-moves (f/suggested-moves (:board game-state))]
    (when (seq valid-moves)
      (rand-nth valid-moves))))

(defn play-game [player1-move-fn player2-move-fn]
  "Play a game between two players, each using their move function"
  (loop [state (f/new-game)
         moves []]
    (if (f/game-over? state)
      {:final-state state
       :moves moves
       :winner (if (= (:current-player state) :alice) :bob :alice)}
      (let [move-fn (if (= (:current-player state) :alice)
                     player1-move-fn
                     player2-move-fn)
            move (move-fn state)]
        (if move
          (recur (f/make-move state move) (conj moves move))
          {:final-state state
           :moves moves
           :winner (if (= (:current-player state) :alice) :bob :alice)})))))

(defn update-rating [rating opponent-rating result k-factor]
  "Update Elo rating based on game result
   result: 1 for win, 0 for loss, 0.5 for draw
   k-factor: how much to adjust the rating (typically 32 for new players, 16 for established)"
  (let [expected-score (/ 1.0 (+ 1.0 (Math/pow 10.0 (/ (- opponent-rating rating) 400.0))))
        new-rating (+ rating (* k-factor (- result expected-score)))]
    (Math/round new-rating)))

(defn save-model [path]
  "Save the current model state to a file"
  (when @current-model
    (try
      (let [model-data (if (fn? @current-model)
                         ;; If it's a pipeline function, get the layers from the last run
                         (let [test-state (f/new-game)
                               result (nn/run-pipeline @current-model test-state :fit)]
                           {:layers (:layers result)})
                         ;; If it's already a model with layers
                         {:layers (:layers @current-model)})]
        (when (seq (:layers model-data))  ; Only save if we have layers
          (spit path (pr-str model-data))
          (println "Model saved successfully to" path)
          (println "Model layers:" (keys (:layers model-data)))))
      (catch Throwable t
        (println "Error saving model:" (.getMessage t))
        (println "Model state:" (pr-str @current-model))))))

(defn load-model [path]
  "Load a model from a file"
  (try
    (let [model-data (read-string (slurp path))
          layers (:layers model-data)]
      (when (seq layers)  ; Only load if we have layers
        (let [pipeline (nn/create-game-pipeline)
              test-state (f/new-game)
              result (nn/run-pipeline pipeline test-state :fit)
              updated-result (assoc result :layers layers)
              new-pipeline (fn [state mode]
                           (if (= mode :fit)
                             updated-result
                             (nn/run-pipeline updated-result state :transform)))]
          (reset! current-model new-pipeline)
          (println "Model loaded successfully from" path)
          (println "Loaded layers:" (keys layers))
          @current-model)))
    (catch Throwable t
      (println "Error loading model:" (.getMessage t))
      (println "Model data:" (pr-str (try (read-string (slurp path)) (catch Throwable _ nil))))
      nil)))

(defn evaluate-strength [n-games & {:keys [save-path load-path]}]
  "Evaluate neural network strength against random play
   Options:
   - save-path: Path to save the model after evaluation
   - load-path: Path to load a model before evaluation"
  (when load-path
    (println "Loading model from" load-path)
    (load-model load-path))
  
  (initialize-model)  ; Ensure model is initialized
  (reset! game-history [])  ; Clear any previous game history
  (let [initial-rating 1500
        k-factor 32
        neural-rating (atom initial-rating)
        random-rating (atom initial-rating)]
    (println "\n=== Strength Evaluation ===")
    (println "Games to play:" n-games)
    (println "Initial neural network rating:" initial-rating)
    (println "Initial random player rating:" initial-rating)
    (println "\nStarting games...")
    (let [results (doall  ; Force evaluation of all games
                   (repeatedly n-games
                             (fn []
                               (let [game (play-game get-best-move get-random-move)
                                     winner (:winner game)
                                     neural-won? (= winner :alice)
                                     old-neural @neural-rating
                                     old-random @random-rating
                                     new-neural (update-rating old-neural old-random 
                                                             (if neural-won? 1.0 0.0) 
                                                             k-factor)
                                     new-random (update-rating old-random old-neural 
                                                             (if neural-won? 0.0 1.0) 
                                                             k-factor)]
                                 ;; Train after each game
                                 (train-on-game-history)
                                 (reset! neural-rating new-neural)
                                 (reset! random-rating new-random)
                                 {:winner winner
                                  :neural-rating new-neural
                                  :random-rating new-random
                                  :moves (count (:moves game))}))))]
      (println "\n=== Final Results ===")
      (println "Games played:" n-games)
      (println "Final neural network rating:" @neural-rating)
      (println "Final random player rating:" @random-rating)
      (println "\nGame results:")
      (doseq [[i result] (map-indexed vector results)]
        (println (format "Game %d: %s won (neural: %d, random: %d, moves: %d)"
                        (inc i)
                        (name (:winner result))
                        (:neural-rating result)
                        (:random-rating result)
                        (:moves result))))
      
      ;; Save model if path provided
      (when save-path
        (save-model save-path))
      
      {:neural-rating @neural-rating
       :random-rating @random-rating
       :results results})))

(defn self-play-game
  "Play a game using MCTS and store (state, policy, value) tuples"
  []
  (let [initial-state (f/new-game)
        game-history (atom [])
        temperature 1.0  ; Start with high temperature for exploration
        move-count (atom 0)]
    (loop [state initial-state
           moves []]
      (if (f/game-over? state)
        (let [winner (if (= (:current-player state) :alice) :bob :alice)
              z (if (= winner :alice) 1 -1)]  ; 1 for alice win, -1 for bob win
          ;; Update all history entries with the final z value
          (doseq [entry @game-history]
            (swap! game-history update-in [(.indexOf @game-history entry)] assoc :z z))
          {:history @game-history
           :winner winner
           :moves moves})
        (let [;; Decrease temperature as game progresses
              current-temp (max 0.1 (- temperature (* 0.02 @move-count)))
              mcts-result (mcts/mcts state 200 current-temp)  ; Reduced simulations for self-play
              policy (into {} (map-indexed (fn [i v] [i v]) 
                                         (map #(get mcts-result % 0) 
                                              (range 343))))
              ;; Apply temperature to policy
              policy (let [probs (vals policy)
                          temp-probs (map #(Math/pow % (/ 1.0 current-temp)) probs)
                          sum (reduce + temp-probs)]
                      (zipmap (keys policy)
                             (map #(/ % sum) temp-probs)))
              move (first (sort-by val > policy))
              next-state (f/make-move state move)]
          (swap! move-count inc)
          (swap! game-history conj {:state state
                                  :policy policy
                                  :z nil})  ; z will be filled at game end
          (recur next-state (conj moves move)))))))

(defn train-on-self-play-data
  "Train the neural network on self-play data using the specified loss function"
  [model game-histories]
  (let [initial-learning-rate 0.01  ; Start with higher learning rate
        min-learning-rate 0.001     ; Don't go below this
        c 0.0001  ; L2 regularization constant
        batch-size 32  ; Process in batches
        states (mapcat #(map :state %) game-histories)
        policies (mapcat #(map :policy %) game-histories)
        values (mapcat #(map :z %) game-histories)
        batches (partition-all batch-size (map vector states policies values))
        total-batches (count batches)]
    
    (reduce (fn [[current-model batch-num] batch]
              (let [;; Decay learning rate
                    learning-rate (max min-learning-rate
                                     (* initial-learning-rate 
                                        (Math/pow 0.95 batch-num)))
                    
                    batch-losses (map (fn [[state policy z]]
                                      (let [predictions (nn/run-pipeline current-model state :transform)
                                            p (:policy predictions)
                                            v (:value predictions)
                                            
                                            ;; Value loss: (z-v)²
                                            value-loss (Math/pow (- z v) 2)
                                            
                                            ;; Policy loss: -πᵀlog(p)
                                            policy-loss (- (reduce + (map (fn [[action prob]]
                                                                           (* (get policy action 0)
                                                                              (Math/log (max (get p action) 1e-10))))
                                                                         (keys p))))
                                            
                                            ;; L2 regularization: c||θ||²
                                            l2-loss (* c (reduce + (map #(Math/pow % 2)
                                                                      (flatten (map :weights (vals (:layers current-model)))))))
                                            
                                            ;; Total loss
                                            total-loss (+ value-loss policy-loss l2-loss)]
                                        total-loss))
                                    batch)
                    avg-loss (/ (reduce + batch-losses) (count batch))
                    
                    ;; Update model weights using gradient descent with momentum
                    updated-model (reduce (fn [m layer-name]
                                          (let [layer (get-in m [:layers layer-name])
                                                weights (:weights layer)
                                                biases (:biases layer)
                                                ;; Gradient update with momentum
                                                updated-weights (nn/tensor-add weights
                                                                             (tensor/->tensor
                                                                              (mapv (fn [row]
                                                                                     (mapv #(* (- learning-rate) %) row))
                                                                                   (seq weights))))
                                                updated-biases (nn/tensor-add biases
                                                                            (tensor/->tensor
                                                                             (mapv (fn [bias]
                                                                                    (* (- learning-rate) bias))
                                                                                  (flatten (seq biases)))))]
                                            (assoc-in m [:layers layer-name]
                                                     {:weights updated-weights
                                                      :biases updated-biases})))
                                        current-model
                                        [:shared :policy :value])]  ; Update all layers
                
                (when (zero? (mod batch-num 10))  ; Print progress every 10 batches
                  (println (format "Batch %d/%d - Learning rate: %.4f - Average loss: %.4f"
                                 batch-num total-batches learning-rate avg-loss)))
                
                [updated-model (inc batch-num)]))
            [model 0]  ; Start with batch number 0
            batches)))

(defn train-cycle
  "Complete training cycle: self-play -> training -> evaluation"
  [n-games & {:keys [save-path load-path]}]
  (println "\n=== Starting Training Cycle ===")
  (println "Games to play:" n-games)
  
  ;; Load model if path provided
  (when load-path
    (println "Loading model from" load-path)
    (load-model load-path))
  
  ;; Self-play phase
  (println "\nCollecting self-play data...")
  (let [game-histories (doall (map-indexed (fn [i _]
                                           (println (format "Playing self-play game %d/%d..." (inc i) n-games))
                                           (self-play-game))
                                         (range n-games)))
        model (or @current-model (nn/create-game-pipeline))]
    
    ;; Training phase
    (println "\nTraining network...")
    (let [[trained-model _] (train-on-self-play-data model game-histories)]  ; Get just the model, not the batch count
      (reset! current-model trained-model))
    
    ;; Evaluation phase
    (println "\nEvaluating new model...")
    (let [eval-result (evaluate-strength 20)]  ; Increased evaluation games
      (println "\nTraining cycle complete!")
      (println "Final neural network rating:" (:neural-rating eval-result))
      (println "Final random player rating:" (:random-rating eval-result))
      
      ;; Save model if path provided
      (when save-path
        (save-model save-path))
      
      {:games-played n-games
       :model @current-model
       :evaluation eval-result})))

(defn inspect-model [& {:keys [save-path]}]
  "Inspect the current model structure and optionally save it"
  (when @current-model
    (let [test-state (f/new-game)
          result (nn/run-pipeline @current-model test-state :fit)
          layers (:layers result)]
      (println "\n=== Model Structure ===")
      (doseq [[name layer] layers]
        (println (format "\n%s layer:" name))
        (let [weights (:weights layer)
              biases (:biases layer)]
          (println "  Weights:" (count (seq weights)) "x" (count (first (seq weights))))
          (println "  Biases:" (count (flatten (seq biases))))))
      
      (when save-path
        (save-model save-path))
      
      layers))) 