(ns latin-squares7.nn-metamorph
  (:require [latin-squares7.functions :as f]
            [latin-squares7.mcts :as mcts]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.tensor :as tensor]
            [tech.v3.datatype.functional :as df]
            [scicloj.metamorph.core :as morph]
            [scicloj.metamorph.ml :as ml]
            [fastmath.core :as fm]
            [fastmath.vector :as fv]
            [fastmath.matrix :as fmx])
  (:import [org.apache.commons.math3.linear Array2DRowRealMatrix RealMatrix]))

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
    (println (format "%s: dims=%s" name dims))
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

;; Neural Network Pipeline Components

(defn board->features-pipe
  "Convert board state to feature tensor"
  []
  (fn [{:metamorph/keys [data mode] :as ctx}]
    (let [board-data (cond
                      (map? data) (if (contains? data :board)
                                   (:board data)
                                   (throw (ex-info "Game state map missing :board key" 
                                                 {:keys (keys data)})))
                      (vector? data) data
                      :else (throw (ex-info "Invalid input data type" 
                                          {:type (type data)
                                           :mode mode})))
          flattened-data (vec (map #(if (nil? %) 0.0 %) 
                                  (flatten board-data)))
          features (debug-tensor "features" (ensure-2d-tensor (vec (take 49 flattened-data))))]
      (assoc ctx :metamorph/data features))))

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
             {:policy (zipmap (range 343) policy-probs)  ; Map move indices to probabilities
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
          move-keys (map mcts/compress-move moves)
          policy-logits (vec (flatten (seq (get predictions :policy))))
          policy-probs (vec (softmax policy-logits))  ; Ensure policy-probs is a vector
          value (get predictions :value)
          
          ;; Filter policy to only include valid moves and normalize
          valid-policy (when (and (seq moves) (seq policy-probs))
                        (let [move-probs (map (fn [key]
                                              (when (< key (count policy-probs))
                                                (nth policy-probs key)))
                                            move-keys)
                              valid-probs (remove nil? move-probs)
                              max-prob (apply max valid-probs)
                              ;; Scale probabilities to be more reasonable
                              scaled-probs (map #(/ % max-prob) valid-probs)
                              total-prob (reduce + scaled-probs)]
                          (if (pos? total-prob)
                            (zipmap moves
                                   (map #(/ % total-prob) scaled-probs))
                            (zipmap moves (repeat (/ 1.0 (count moves)))))))]
      
      (assoc ctx :metamorph/data
             {:policy (or valid-policy {})
              :value (or value 0.0)}))))

;; Pipeline Creation

(defn create-game-pipeline []
  "Create a complete pipeline for the game"
  (morph/pipeline
   (board->features-pipe)
   (create-layer-pipe)
   (forward-pass-pipe)
   (predict-pipe)))

;; Pipeline Operations

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

(defn train-model [n-games]
  "Train the model using our pipeline"
  (let [pipeline (create-game-pipeline)
        games (repeatedly n-games mcts/auto-play-full-game)
        trained-model (reduce (fn [model game]
                              (try
                                (println "\n=== Training Step ===")
                                (println "Game state:")
                                (f/print-board (:board game))
                                
                                ;; Validate model state
                                (when-not (map? model)
                                  (throw (ex-info "Invalid model state" 
                                                {:model-type (type model)})))
                                
                                ;; Run training step
                                (let [result (run-pipeline model game :fit)]
                                  ;; Validate result
                                  (when-not (map? result)
                                    (throw (ex-info "Invalid training result" 
                                                  {:result-type (type result)})))
                                  
                                  ;; Print predictions if available
                                  (when-let [policy (:policy result)]
                                    (println "\nPolicy predictions (top 5):")
                                    (doseq [[move prob] (take 5 (sort-by val > policy))]
                                      (println (format "Move %s: %.6f" move prob))))
                                  (when-let [value (:value result)]
                                    (println (format "Value prediction: %.6f" value)))
                                  
                                  ;; Return the model for next iteration
                                  (if (contains? result :layers)
                                    result  ; If result contains layers, it's a model
                                    model))  ; Otherwise keep using previous model
                                
                                (catch Exception e
                                  (println "\nError during training:")
                                  (println "Exception type:" (type e))
                                  (println "Exception message:" (.getMessage e))
                                  (println "Stack trace:")
                                  (.printStackTrace e)
                                  model)))  ; Return unchanged model on error
                            pipeline
                            games)]
    ;; Final validation
    (when-not (map? trained-model)
      (throw (ex-info "Training failed to produce valid model" 
                     {:model-type (type trained-model)})))
    trained-model))

(defn get-best-move [game-state]
  "Get the best move using the trained model"
  (let [trained-model (train-model 10)  ; Train on 10 games first
        result (run-pipeline trained-model game-state :transform)]
    (when (and (map? result)
               (seq (get result :policy)))
      (key (apply max-key val (get result :policy))))))

(defn autoplay-from-position [game-state max-moves]
  "Autoplay from a given position using the neural network model"
  (loop [state game-state
         moves-made 0
         moves []]
    (if (or (>= moves-made max-moves)
            (f/game-over? state))
      {:final-state state
       :moves-made moves-made
       :solved? (f/solved? state)
       :moves moves}
      (let [move (get-best-move state)]
        (if move
          (recur (f/make-move state move)
                 (inc moves-made)
                 (conj moves move))
          {:final-state state
           :moves-made moves-made
           :solved? (f/solved? state)
           :moves moves})))))