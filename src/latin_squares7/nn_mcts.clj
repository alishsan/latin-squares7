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
                  :bob  ; If it's alice's turn and game is over, bob won
                  :alice))]  ; If it's bob's turn and game is over, alice won
    (println "\n=== Game Summary ===")
    (println "Final board state:")
    (f/print-board (:board final-state))
    (println "\nGame statistics:")
    (println "Moves made:" (:moves-made result))
    (println "Game over:" (:game-over? result))
    (when winner
      (println "Winner:" (name winner)))
    (println "\nMove history:")
    (doseq [[i move] (map-indexed vector (:moves result))]
      (println (format "Move %d: %s" (inc i) move)))
    result))

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

(defn evaluate-strength [n-games]
  "Evaluate neural network strength against random play"
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
      {:neural-rating @neural-rating
       :random-rating @random-rating
       :results results}))) 