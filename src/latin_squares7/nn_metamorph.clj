(ns latin-squares7.nn-metamorph
  (:require [latin-squares7.functions :as f]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.tensor :as tensor]
            [tech.v3.datatype.functional :as df]
            [scicloj.metamorph.core :as morph]
            [scicloj.metamorph.ml :as ml]
            [fastmath.core :as fm]
            [fastmath.vector :as fv]
            [fastmath.matrix :as fmx]))

;; Helper Functions
(defn random-weight [input-size output-size]
  (* (Math/sqrt (/ 2.0 (+ input-size output-size)))
     (- (rand) 0.5)))

(defn create-weight-row [input-size output-size]
  (vec (repeatedly input-size
                  #(random-weight input-size output-size))))

(defn get-tensor-dims [t]
  "Get tensor dimensions by analyzing the tensor structure"
  (if (tensor/tensor? t)
    (let [flat-seq (seq t)
          total-elements (count flat-seq)]
      (if (zero? total-elements)
        [0 0]
        (let [first-row (first flat-seq)
              row-size (if (sequential? first-row)
                        (count first-row)
                        total-elements)
              num-rows (if (sequential? first-row)
                        (count flat-seq)
                        1)]
          [num-rows row-size])))
    (if (sequential? t)
      [(count t) (count (first t))]
      [1 1])))

(defn transpose-matrix [matrix]
  "Transpose a matrix represented as a sequence of sequences"
  (apply mapv vector matrix))

(defn debug-tensor [name t]
  "Print tensor information for debugging"
  (let [dims (get-tensor-dims t)]
    ;; (println (format "%s: dims=%s" name dims))
    t))

(defn ensure-2d-tensor [t]
  "Ensure tensor is 2D by reshaping if necessary"
  (let [t (tensor/->tensor t)
        dims (get-tensor-dims t)]
    (if (= 1 (count dims))
      (tensor/reshape t [(first dims) 1])
      t)))

(defn matrix-multiply [a b]
  "Multiply two matrices using simple tensor operations"
  (let [a (ensure-2d-tensor a)
        b (ensure-2d-tensor b)
        a-dims (get-tensor-dims a)
        b-dims (get-tensor-dims b)]
    (when (not= (second a-dims) (first b-dims))
      (throw (ex-info "Incompatible dimensions for matrix multiplication"
                     {:a-dims a-dims 
                      :b-dims b-dims})))
    (tensor/->tensor
     (for [i (range (first a-dims))]
       (for [j (range (second b-dims))]
         (reduce + (map * (nth (seq a) i)
                          (map #(nth % j) (seq b)))))))))

(defn relu [x]
  "ReLU activation function"
  (max 0.0 x))

(defn sigmoid [x]
  "Sigmoid activation function"
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn softmax [xs]
  "Compute softmax probabilities with numerical stability"
  (let [max-x (apply max xs)
        exp-xs (map #(Math/exp (- % max-x)) xs)  ; Subtract max for numerical stability
        sum-exp (reduce + exp-xs)]
    (map #(/ % sum-exp) exp-xs)))

(defn tensor-add [a b]
  "Add two tensors element-wise"
  (let [a (ensure-2d-tensor a)
        b (ensure-2d-tensor b)
        a-dims (get-tensor-dims a)
        b-dims (get-tensor-dims b)]
    (cond
      ;; If b is a single number, broadcast it
      (and (= 1 (first b-dims)) (= 1 (second b-dims)))
      (tensor/->tensor
       (mapv (fn [a-row]
               (mapv #(+ % (first (flatten (seq b)))) a-row))
             (seq a)))
      
      ;; If a is a single number, broadcast it
      (and (= 1 (first a-dims)) (= 1 (second a-dims)))
      (tensor/->tensor
       (mapv (fn [b-row]
               (mapv #(+ (first (flatten (seq a))) %) b-row))
             (seq b)))
      
      ;; If b is a row vector (1 x n), broadcast it across rows
      (and (= 1 (first b-dims)))
      (tensor/->tensor
       (mapv (fn [a-row]
               (mapv + a-row (first (seq b))))
             (seq a)))
      
      ;; If a is a row vector (1 x n), broadcast it across rows
      (and (= 1 (first a-dims)))
      (tensor/->tensor
       (mapv (fn [b-row]
               (mapv + (first (seq a)) b-row))
             (seq b)))
      
      ;; If b is a column vector (n x 1), broadcast it across columns
      (and (= 1 (second b-dims)))
      (tensor/->tensor
       (mapv (fn [a-row b-val]
               (mapv #(+ % (first (flatten (seq b-val)))) a-row))
             (seq a)
             (mapv vector (flatten (seq b)))))
      
      ;; If a is a column vector (n x 1), broadcast it across columns
      (and (= 1 (second a-dims)))
      (tensor/->tensor
       (mapv (fn [b-row a-val]
               (mapv #(+ (first (flatten (seq a-val))) %) b-row))
             (seq b)
             (mapv vector (flatten (seq a)))))
      
      ;; Otherwise, add element-wise
      :else
      (tensor/->tensor
       (mapv (fn [a-row b-row]
               (mapv + a-row b-row))
             (seq a)
             (seq b))))))

(defn compress-move [[r c n]]
  (+ (* 100 r) (* 10 c) n))

(defn decompress-move [move-int]
  (when move-int
    [(quot move-int 100)
     (quot (mod move-int 100) 10)
     (mod move-int 10)]))

(defn board->features [board]
  "Convert a 7x7 board into a feature vector for the neural network"
  (let [flattened (flatten board)]
    (tensor/->tensor
     (mapv #(if (nil? %) 0.0 %) flattened))))

(defn board->features-pipe []
  "Convert board state to feature vector"
  (fn [{:metamorph/keys [data] :as ctx}]
    (let [board (:board data)
          features (board->features board)]
      (assoc ctx :metamorph/data features))))

;; Neural Network Pipeline Components
(defn create-layer-pipe
  "Create neural network layers with weights and biases"
  []
  (fn [{:metamorph/keys [data mode] :as ctx}]
    (let [input-size 49  ; 7x7 board
          hidden-size 128
          policy-size 343  ; 7x7x7 possible moves
          value-size 1
          
          ;; Shared layers
          shared-weights1 (debug-tensor "shared-weights1" 
                                      (ensure-2d-tensor
                                       (mapv (fn [_]
                                              (create-weight-row input-size hidden-size))
                                            (range hidden-size))))
          shared-biases1 (debug-tensor "shared-biases1" 
                                     (ensure-2d-tensor (repeat hidden-size 0.0)))
          
          ;; Policy head
          policy-weights (debug-tensor "policy-weights" 
                                     (ensure-2d-tensor
                                      (mapv (fn [_]
                                             (create-weight-row hidden-size policy-size))
                                           (range policy-size))))
          policy-biases (debug-tensor "policy-biases" 
                                    (ensure-2d-tensor (repeat policy-size 0.0)))
          
          ;; Value head
          value-weights (debug-tensor "value-weights" 
                                    (ensure-2d-tensor
                                     (mapv (fn [_]
                                            (create-weight-row hidden-size value-size))
                                          (range value-size))))
          value-biases (debug-tensor "value-biases" 
                                   (ensure-2d-tensor (repeat value-size 0.0)))]
      
      (-> ctx
          (assoc :metamorph/data data)
          (assoc :layers {:shared {:weights shared-weights1
                                  :biases shared-biases1}
                         :policy {:weights policy-weights
                                 :biases policy-biases}
                         :value {:weights value-weights
                                :biases value-biases}})))))

(defn forward-pass-pipe
  "Perform forward pass through the network"
  []
  (fn [{:metamorph/keys [data mode] :as ctx}]
    (let [layers (:layers ctx)
          input (debug-tensor "input" (ensure-2d-tensor data))
          input-transposed (debug-tensor "input-transposed" 
                                       (tensor/->tensor (mapv vector (flatten (seq input)))))
          
          ;; Shared layers
          shared-weights (debug-tensor "shared-weights" (:weights (:shared layers)))
          shared-biases (debug-tensor "shared-biases" (ensure-2d-tensor (:biases (:shared layers))))
          
          ;; Forward pass through shared layers
          shared-out (debug-tensor "shared-out" (matrix-multiply shared-weights input-transposed))
          shared-out-with-bias (debug-tensor "shared-out-with-bias" 
                                           (tensor-add shared-out 
                                                      (tensor/->tensor (mapv vector (flatten (seq shared-biases))))))
          shared-activation (debug-tensor "shared-activation" 
                                        (tensor/->tensor (mapv relu (flatten (seq shared-out-with-bias)))))
          shared-activation-transposed (debug-tensor "shared-activation-transposed"
                                                  (tensor/->tensor (mapv vector (flatten (seq shared-activation)))))
          
          ;; Policy head
          policy-weights (debug-tensor "policy-weights" (:weights (:policy layers)))
          policy-biases (debug-tensor "policy-biases" (ensure-2d-tensor (:biases (:policy layers))))
          
          ;; Policy head forward pass
          policy-out (debug-tensor "policy-out" (matrix-multiply policy-weights shared-activation-transposed))
          policy-out-with-bias (debug-tensor "policy-out-with-bias" 
                                           (tensor-add policy-out 
                                                      (tensor/->tensor (mapv vector (flatten (seq policy-biases))))))
          policy-logits (vec (flatten (seq policy-out-with-bias)))
          policy-probs (softmax policy-logits)
          
          ;; Value head
          value-weights (debug-tensor "value-weights" (:weights (:value layers)))
          value-biases (debug-tensor "value-biases" (ensure-2d-tensor (:biases (:value layers))))
          
          ;; Value head forward pass
          value-out (debug-tensor "value-out" (matrix-multiply value-weights shared-activation-transposed))
          value-out-with-bias (debug-tensor "value-out-with-bias" 
                                          (tensor-add value-out 
                                                     (tensor/->tensor (mapv vector (flatten (seq value-biases))))))
          value (sigmoid (first (flatten (seq value-out-with-bias))))]
      
      (assoc ctx :metamorph/data
             {:policy (zipmap (map f/compress-move (for [r (range 7)
                                                       c (range 7)
                                                       n (range 1 8)]
                                                   [r c n]))
                            policy-probs)  ; Map compressed move integers to probabilities
              :value value}))))

(defn predict-pipe
  "Make predictions using the trained model"
  []
  (fn [{:metamorph/keys [data mode] :as ctx}]
    (let [predictions (if (map? data)
                       data
                       (throw (ex-info "Expected map data in predict-pipe" 
                                     {:data-type (type data)})))
          game-state (:metamorph/data (get-in ctx [:metamorph/context :original-data]))
          moves (f/suggested-moves (:board game-state))
          move-keys (map f/compress-move moves)  ; Use the same compression as functions.clj
          policy-logits (vec (flatten (seq (get predictions :policy))))
          policy-probs (vec (softmax policy-logits))  ; Ensure policy-probs is a vector
          value (get predictions :value)
          
          ;; Filter policy to only include valid moves and normalize
          valid-policy (when (and (seq moves) (seq policy-probs))
                        (let [move-probs (map (fn [move]
                                              (let [compressed (f/compress-move move)
                                                    idx (+ (* 7 (first move))
                                                         (* 49 (second move))
                                                         (dec (nth move 2)))]
                                                (when (< idx (count policy-probs))
                                                  (nth policy-probs idx))))
                                            moves)
                              valid-probs (remove nil? move-probs)
                              max-prob (apply max valid-probs)
                              ;; Scale probabilities to be more reasonable
                              scaled-probs (map #(/ % max-prob) valid-probs)
                              total-prob (reduce + scaled-probs)]
                          (if (pos? total-prob)
                            (zipmap moves  ; Use move vectors directly as keys
                                   (map #(/ % total-prob) scaled-probs))
                            (zipmap moves  ; Use move vectors directly as keys
                                   (repeat (/ 1.0 (count moves)))))))]
      
      (assoc ctx :metamorph/data
             {:policy (or valid-policy {})
              :value (or value 0.0)}))))

;; Pipeline Creation and Operations
(defn create-game-pipeline []
  "Create a complete pipeline for the game"
  (morph/pipeline
   (board->features-pipe)
   (create-layer-pipe)
   (forward-pass-pipe)
   (predict-pipe)))

(defn run-pipeline [pipeline data mode]
  "Run a pipeline of functions on the data"
  (let [ctx {:metamorph/data data
             :metamorph/mode mode
             :metamorph/context {:original-data data}}
        result ((morph/pipeline pipeline) ctx)]
    (if (= mode :fit)
      ;; In fit mode, return the model state
      (or (:layers result)  ; Return layers if available
          (:metamorph/data result))  ; Otherwise return the data
      ;; In transform mode, return the predictions
      (:metamorph/data result))))

;; Model state
(def ^:private trained-model (atom nil))

(defn get-trained-model []
  @trained-model)

(defn set-trained-model [model]
  (reset! trained-model model))

(defn save-model [model path]
  "Save the trained model to a file"
  (try
    (let [model-data {:layers (into {} (map (fn [[name layer]]
                                             [name {:weights (vec (map vec (seq (:weights layer))))
                                                   :biases (vec (flatten (seq (:biases layer))))}])
                                           (:layers model)))}]
      (spit path (pr-str model-data))
      (println "Model saved successfully to" path))
    (catch Exception e
      (println "Error saving model:" (.getMessage e)))))

(defn load-model [path]
  "Load a trained model from a file"
  (try
    (let [model-data (read-string (slurp path))
          layers (into {} (map (fn [[name layer]]
                               [name {:weights (tensor/->tensor (:weights layer))
                                     :biases (tensor/->tensor (:biases layer))}])
                             (:layers model-data)))]
      (println "Model loaded successfully from" path)
      (create-game-pipeline))  ; Create a new pipeline with the loaded weights
    (catch Exception e
      (println "Error loading model:" (.getMessage e))
      nil)))

(defn get-best-move [game-state]
  "Get the best move using the trained model"
  (when (nil? @trained-model)
    (throw (ex-info "Model not initialized" {})))
  (let [pipeline @trained-model
        result (run-pipeline pipeline game-state :transform)
        policy (:policy result)
        valid-moves (f/suggested-moves (:board game-state))]
    (when (seq valid-moves)
      (apply max-key #(get policy % 0.0) valid-moves))))

(defn self-play-game []
  "Play a full game using pure neural network and store (state, policy, value) tuples"
  (let [initial-state (f/new-game)
        game-history (atom [])]
    (loop [state initial-state
           moves []
           move-count 0]
      (if (or (f/game-over? state)
              (>= move-count 49))  ; Maximum possible moves in a 7x7 board
        (let [winner (if (= (:current-player state) :alice) :bob :alice)
              z (if (= winner :alice) 1 -1)]  ; 1 for alice win, -1 for bob win
          ;; Update all history entries with the final z value
          (doseq [entry @game-history]
            (swap! game-history update-in [(.indexOf @game-history entry)] assoc :z z))
          {:board (:board state)
           :moves moves
           :history @game-history
           :winner winner
           :moves-made move-count})
        (let [predictions (run-pipeline @trained-model state :transform)
              policy (:policy predictions)
              move (first (sort-by val > policy))  ; Select best move
              next-state (f/make-move state move)]
          (swap! game-history conj {:state state
                                  :policy policy
                                  :z nil})  ; z will be filled at game end
          (recur next-state (conj moves move) (inc move-count)))))))

(defn train-on-self-play [n-games]
  "Train the model through self-play games using AlphaGo-style training"
  (let [pipeline (create-game-pipeline)
        games (repeatedly n-games self-play-game)
        learning-rate 0.01
        c 0.0001]  ; L2 regularization constant
    
    (defn train-on-position [current-model {:keys [state policy z]}]
      (let [predictions (run-pipeline current-model state :transform)
            p (:policy predictions)
            v (:value predictions)
            value-loss (Math/pow (- z v) 2)
            policy-loss (let [valid-actions (filter #(and (get p %)
                                                         (get policy %))
                                                  (keys p))]
                       (if (empty? valid-actions)
                         0.0  ; No valid actions, no policy loss
                         (- (reduce + (map (fn [action]
                                            (let [pred-prob (get p action 0.0)
                                                  target-prob (get policy action 0.0)]
                                              (* target-prob
                                                 (Math/log (max pred-prob 1e-10)))))
                                          valid-actions)))))
            l2-loss (* c (reduce + (map #(Math/pow % 2)
                                     (flatten (map :weights (vals (:layers current-model)))))))
            total-loss (+ value-loss policy-loss l2-loss)]
        
        ;; Debug logging
        (println "\nTraining on position:")
        (println "  State:" state)
        (println "  Policy size:" (count policy))
        (println "  Value prediction:" v)
        (println "  Z value:" z)
        
        ;; Policy debug
        (println "\nPolicy debug:")
        (println "  Policy keys:" (keys p))
        (println "  Policy values:" (vals p))
        (println "  Target policy keys:" (keys policy))
        (println "  Target policy values:" (vals policy))
        
        ;; Validate policy
        (when (or (nil? p) (empty? p))
          (throw (ex-info "Invalid policy predictions" 
                        {:predictions predictions
                         :policy p})))
        
        ;; Loss debug
        (println "\nLosses:")
        (println "    Value loss:" value-loss)
        (println "    Policy loss:" policy-loss)
        (println "    L2 loss:" l2-loss)
        (println "    Total loss:" total-loss)
        
        ;; Update model weights using gradient descent
        (let [updated-model (reduce (fn [m layer-name]
                                    (let [layer (get-in m [:layers layer-name])
                                          weights (:weights layer)
                                          biases (:biases layer)]
                                      (when (and weights biases)  ; Only update if both exist
                                        (let [;; Simple gradient update
                                              updated-weights (tensor-add weights
                                                                         (tensor/->tensor
                                                                          (mapv (fn [row]
                                                                                 (mapv #(* (- learning-rate) %) row))
                                                                               (seq weights))))
                                              updated-biases (tensor-add biases
                                                                        (tensor/->tensor
                                                                         (mapv (fn [bias]
                                                                                (* (- learning-rate) bias))
                                                                              (flatten (seq biases)))))]
                                          (assoc-in m [:layers layer-name]
                                                   {:weights updated-weights
                                                    :biases updated-biases})))))
                                  current-model
                                  [:policy :value])]
          (or updated-model current-model))))  ; Return current model if update failed
    
    (defn train-on-game [model game]
      (try
        (println "\n=== Training on Self-Play Game ===")
        (println "Game result:" (if (= (:winner game) :alice) "Alice won" "Bob won"))
        (reduce train-on-position model (:history game))
        (catch Exception e
          (println "\nError during training:")
          (println "Exception message:" (.getMessage e))
          (println "Stack trace:")
          (.printStackTrace e)
          model)))  ; Return unchanged model on error
    
    (reduce train-on-game pipeline games)))

(defn initialize-model []
  "Initialize the model through self-play training"
  (train-on-self-play 100))

(defn retrain-model [num-games & {:keys [load-path save-path]}]
  "Retrain the model on new self-play games.
   Options:
   - load-path: Path to load a previous model from
   - save-path: Path to save the trained model to"
  (println "Retraining model on" num-games "self-play games...")
  (let [initial-model (if load-path
                       (do
                         (println "Loading previous model from" load-path)
                         (load-model load-path))
                       (create-game-pipeline))
        trained-model (train-on-self-play num-games)]
    (when save-path
      (println "Saving trained model to" save-path)
      (save-model trained-model save-path))
    (set-trained-model trained-model)
    trained-model))

(defn autoplay-from-position [game-state max-moves]
  "Autoplay from a given position using the neural network model"
  (f/autoplay-from-position game-state max-moves get-best-move))

(defn train-on-batch [model batch-data]
  "Train the model on a batch of positions"
  (let [learning-rate 0.01
        batch-size (count batch-data)
        inputs (map :input batch-data)
        policy-targets (map :policy-target batch-data)
        value-targets (map :value-target batch-data)
        forward-results (map #(run-pipeline model {:board %} :transform) inputs)
        policy-losses (map (fn [result target]
                           (let [pred-policy (:policy result)
                                 target-policy target
                                 valid-moves (keys target-policy)
                                 pred-probs (map #(get pred-policy % 0.0) valid-moves)
                                 target-probs (map #(get target-policy %) valid-moves)]
                             (- (reduce + (map (fn [p t] (- (* t (Math/log (max p 1e-10)))))
                                             pred-probs target-probs)))))
                         forward-results policy-targets)
        value-losses (map (fn [result target]
                          (let [pred-value (:value result)]
                            (Math/pow (- pred-value target) 2)))
                        forward-results value-targets)
        avg-policy-loss (/ (reduce + policy-losses) batch-size)
        avg-value-loss (/ (reduce + value-losses) batch-size)
        update-layer (fn [layer]
                      {:weights (tensor-add (:weights layer)
                                          (tensor/->tensor
                                           (mapv (fn [row]
                                                 (mapv #(* (- learning-rate) %) row))
                                               (seq (:weights layer)))))
                       :biases (tensor-add (:biases layer)
                                         (tensor/->tensor
                                          (mapv (fn [bias]
                                                (* (- learning-rate) bias))
                                              (flatten (seq (:biases layer))))))})
        updated-model (reduce (fn [current-model [input _ _]]
                              (let [result (run-pipeline current-model {:board input} :fit)
                                    layers (:layers result)
                                    updated-layers (into {} (map (fn [[name layer]]
                                                                [name (update-layer layer)])
                                                              layers))]
                                (assoc result :layers updated-layers)))
                            model
                            (map vector inputs policy-targets value-targets))]
    (println "Training batch stats:")
    (println "Average policy loss:" avg-policy-loss)
    (println "Average value loss:" avg-value-loss)
    updated-model))
