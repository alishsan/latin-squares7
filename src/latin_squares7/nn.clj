(ns latin-squares7.nn
  (:require [latin-squares7.functions :as f]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.tensor :as tensor]
            [tech.v3.datatype.functional :as df]
            [scicloj.metamorph.core :as morph]
            [scicloj.metamorph.ml :as ml]
            [fastmath.core :as fm]
            [fastmath.vector :as fv]
            [fastmath.matrix :as fmx])
  (:import [org.apache.commons.math3.linear Array2DRowRealMatrix RealMatrix]))

;; Model state
(def ^:private trained-model (atom nil))

(defn get-trained-model []
  @trained-model)

(defn set-trained-model [model]
  (reset! trained-model model))

;; Helper Functions
(defn get-random-move [game-state]
  "Get a random valid move from the current position"
  (let [valid-moves (f/valid-moves (:board game-state))]
    (when (seq valid-moves)
      (rand-nth valid-moves))))

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

(defn to-double [x]
  "Convert a value to double, handling nil and other types"
  (cond
    (nil? x) 0.0
    (number? x) (double x)
    :else (throw (ex-info "Cannot convert to double" {:value x}))))

(defn seq->tensor [x]
  "Convert a sequence to a tensor, handling various input types"
  (cond
    (nil? x) (tensor/->tensor [[0.0]])
    (number? x) (tensor/->tensor [[(to-double x)]])
    (vector? x) (if (number? (first x))
                  (tensor/->tensor [(mapv to-double x)])
                  (tensor/->tensor (mapv #(mapv to-double %) x)))
    (seq? x) (if (number? (first x))
               (tensor/->tensor [(mapv to-double (vec x))])
               (tensor/->tensor (mapv #(mapv to-double %) (vec x))))
    :else (throw (ex-info "Cannot convert to tensor" {:value x}))))

(defn ensure-tensor [x]
  "Convert a value to a tensor if it isn't already"
  (cond
    (tensor/tensor? x) x
    :else (seq->tensor x)))

(defn tensor-add [a b]
  "Add tensors with broadcasting"
  (let [a (ensure-2d-tensor a)
        b (ensure-2d-tensor b)]
    (tensor/->tensor
     (for [i (range (first (get-tensor-dims a)))]
       (for [j (range (second (get-tensor-dims a)))]
         (double (+ (if (number? (nth (seq a) i))
                     (nth (seq a) i)
                     (nth (vec (seq (nth (seq a) i))) j))
                   (if (number? (nth (seq b) i))
                     (nth (seq b) i)
                     (nth (vec (seq (nth (seq b) i))) j)))))))))

(defn tensor-multiply [a b]
  "Multiply tensors with broadcasting using df/*"
  (df/* (ensure-tensor a) (ensure-tensor b)))

(defn transpose [tensor]
  "Transpose a tensor using our own transpose-matrix function"
  (tensor/->tensor (transpose-matrix (seq (ensure-tensor tensor)))))

(defn matrix-multiply [a b]
  "Multiply two matrices using tensor operations"
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
         (double (reduce + (map #(double (* %1 %2))
                               (if (number? (nth (seq a) i))
                                 [(nth (seq a) i)]
                                 (vec (seq (nth (seq a) i))))
                               (map #(if (number? %)
                                     %
                                     (nth (vec (seq %)) j))
                                  (seq b))))))))))

(defn sigmoid [x]
  "Sigmoid activation function"
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn sigmoid-derivative [x]
  "Derivative of sigmoid function: f'(x) = f(x) * (1 - f(x))"
  (* x (- 1.0 x)))

(defn relu [x]
  "ReLU activation function"
  (max 0.0 x))

(defn relu-derivative [x]
  "Derivative of ReLU function"
  (if (> x 0.0) 1.0 0.0))

(defn softmax [xs]
  "Compute softmax probabilities with numerical stability"
  (let [max-x (apply max xs)
        exp-xs (map #(Math/exp (- % max-x)) xs)  ; Subtract max for numerical stability
        sum-exp (reduce + exp-xs)]
    (map #(/ % sum-exp) exp-xs)))

(defn tensor-map [f tensor]
  "Apply a function to each element of a tensor"
  (let [tensor (ensure-tensor tensor)
        data (seq tensor)]
    (tensor/->tensor
     (if (number? (first data))
       (map f data)  ; 1D tensor
       (map #(map f %) data)))))  ; 2D tensor

(defn forward-pass [network input]
  "Perform a forward pass through the network"
  (loop [layers (:layers network)
         activations [(ensure-tensor input)]
         zs []]
    (if (empty? layers)
      {:activations (vec activations)
       :zs (vec zs)}
      (let [layer (first layers)
            weights (ensure-2d-tensor (:weights layer))
            biases (ensure-2d-tensor (:biases layer))
            last-activation (ensure-2d-tensor (last activations))
            z (tensor-add (matrix-multiply weights last-activation)
                         biases)
            activation (tensor-map sigmoid z)]
        (recur (rest layers)
               (conj activations activation)
               (conj zs z))))))

(defn safe-nth [coll i]
  "Safely get the nth element of a collection, returning nil if out of bounds"
  (try
    (nth coll i)
    (catch IndexOutOfBoundsException e
      nil)))

(defn safe-get-tensor-element [tensor i j]
  "Safely get an element from a tensor, handling both scalar and vector cases"
  (let [data (seq tensor)]
    (if (nil? data)
      0.0
      (let [row (safe-nth data i)]
        (if (number? row)
          row
          (let [col (safe-nth (vec (seq row)) j)]
            (if (nil? col) 0.0 (double col))))))))

(defn backward-pass [network forward-result target]
  "Perform backpropagation to update weights"
  (let [activations (:activations forward-result)
        zs (:zs forward-result)
        layers (:layers network)
        learning-rate (to-double (:learning-rate network))
        
        ;; Calculate output layer error
        output-activation (last activations)
        target-tensor (ensure-tensor target)
        output-error (tensor/->tensor
                     (for [i (range (first (get-tensor-dims output-activation)))]
                       (for [j (range (second (get-tensor-dims output-activation)))]
                         (double (- (safe-get-tensor-element output-activation i j)
                                  (safe-get-tensor-element target-tensor i j))))))
        output-delta (tensor/->tensor
                     (for [i (range (first (get-tensor-dims output-error)))]
                       (for [j (range (second (get-tensor-dims output-error)))]
                         (double (* (safe-get-tensor-element output-error i j)
                                  (sigmoid-derivative
                                   (safe-get-tensor-element output-activation i j)))))))
        
        ;; Backpropagate error
        deltas (reduce (fn [deltas [layer z activation prev-activation]]
                        (let [weight-delta (matrix-multiply (first deltas)
                                                          (transpose prev-activation))
                              bias-delta (first deltas)
                              error (matrix-multiply (transpose (:weights layer))
                                                   (first deltas))
                              delta (tensor/->tensor
                                    (for [i (range (first (get-tensor-dims error)))]
                                      (for [j (range (second (get-tensor-dims error)))]
                                        (double (* (safe-get-tensor-element error i j)
                                                 (sigmoid-derivative
                                                  (safe-get-tensor-element activation i j)))))))]
                          (conj deltas delta)))
                      [output-delta]
                      (map vector
                           (reverse (rest layers))
                           (reverse (rest zs))
                           (reverse (rest activations))
                           (reverse (butlast activations))))
        
        ;; Update weights and biases
        new-layers (map (fn [layer delta activation prev-activation]
                         (let [weight-update (tensor/->tensor
                                            (for [i (range (first (get-tensor-dims delta)))]
                                              (for [j (range (second (get-tensor-dims delta)))]
                                                (double (* learning-rate
                                                         (safe-get-tensor-element delta i j))))))
                               bias-update (tensor/->tensor
                                          (for [i (range (first (get-tensor-dims delta)))]
                                            (for [j (range (second (get-tensor-dims delta)))]
                                              (double (* learning-rate
                                                       (safe-get-tensor-element delta i j))))))
                               new-weights (tensor/->tensor
                                          (for [i (range (first (get-tensor-dims (:weights layer))))]
                                            (for [j (range (second (get-tensor-dims (:weights layer))))]
                                              (double (- (safe-get-tensor-element (:weights layer) i j)
                                                       (safe-get-tensor-element weight-update i j))))))
                               new-biases (tensor/->tensor
                                         (for [i (range (first (get-tensor-dims (:biases layer))))]
                                           (for [j (range (second (get-tensor-dims (:biases layer))))]
                                             (double (+ (safe-get-tensor-element (:biases layer) i j)
                                                      (safe-get-tensor-element bias-update i j))))))]
                           {:weights new-weights
                            :biases new-biases}))
                       (reverse layers)
                       (reverse deltas)
                       (reverse (rest activations))
                       (reverse (butlast activations)))]
    (assoc network :layers (vec (reverse new-layers)))))

(defn board->features [board]
  "Convert a 7x7 board into a feature vector for the neural network"
  (let [flattened (flatten board)]
    (tensor/->tensor
     (mapv #(if (nil? %) 0.0 %) flattened))))

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
          features (tensor/->tensor (vec (take 49 flattened-data)))]
      (assoc ctx :metamorph/data (vec (map double (flatten (seq features))))))))

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
          moves (f/valid-moves (:board game-state))
          policy-logits (vec (flatten (seq (get predictions :policy))))
          policy-probs (vec (softmax policy-logits))  ; Ensure policy-probs is a vector
          value (get predictions :value)
          
          ;; Filter policy to only include valid moves and normalize
          valid-policy (when (and (seq moves) (seq policy-probs))
                        (let [move-probs (map (fn [move]
                                              (let [move-key (f/compress-move move)]
                                                (when (< move-key (count policy-probs))
                                                  (nth policy-probs move-key))))
                                            moves)
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

;; Neural Network Implementation
(defn initialize-weights [input-size output-size]
  "Initialize weights using Xavier/Glorot initialization"
  (let [scale (Math/sqrt (/ 2.0 (+ input-size output-size)))
        random-weight #(* scale (- (rand) 0.5))
        create-row #(tensor/->tensor (repeatedly input-size random-weight))]
    (tensor/->tensor (repeatedly output-size create-row))))

(defn initialize-biases [size]
  (tensor/->tensor (repeat size 0.0)))

(defn create-layer [input-size output-size]
  {:weights (initialize-weights input-size output-size)
   :biases (initialize-biases output-size)})

(defn create-network [layer-sizes]
  "Create a neural network with the given layer sizes"
  (let [layers (map-indexed
                (fn [i size]
                  (let [input-size (nth layer-sizes i)
                        output-size (nth layer-sizes (inc i))]
                    (create-layer input-size output-size)))
                (range (dec (count layer-sizes))))]
    {:layers layers
     :learning-rate 0.01}))

(defn get-tensor-shape [tensor]
  "Get the shape of a tensor using available operations"
  (if (number? tensor)
    [1]  ; Scalar is a 0D tensor, but we'll treat it as 1D
    (let [data (seq tensor)]
      (if (nil? data)
        [1]  ; Empty tensor
        (let [first-elem (first data)]
          (if (number? first-elem)
            [(count data)]  ; 1D tensor
            [(count data) (count first-elem)]))))))  ; 2D tensor

(defn get-tensor-dims [tensor]
  "Get tensor dimensions, handling both tensor and scalar cases"
  (let [shape (get-tensor-shape tensor)]
    (case (count shape)
      0 [1 1]  ; Scalar
      1 (let [len (first shape)]
          (if (= len 1)
            [1 1]  ; Single element
            [len 1]))  ; Column vector
      2 shape  ; 2D tensor
      [1 1])))  ; Fallback

(defn tensor-select [tensor dims idx]
  "Safely select from a tensor based on its dimensions"
  (let [[rows cols] dims
        shape (get-tensor-shape tensor)]
    (cond
      (number? tensor) tensor  ; If it's a scalar, return as is
      (= (count shape) 1) (if (= idx 0)
                           tensor  ; Single element tensor
                           (nth (seq tensor) idx))  ; 1D tensor
      :else (let [data (seq tensor)
                  row (nth data idx)]
              (if (number? row)
                [row]  ; Single element row
                (tensor/->tensor row))))))  ; Convert row to tensor

(defn broadcast-shapes [a b]
  "Ensure tensors have compatible shapes for operations"
  (let [a-tensor (ensure-tensor a)
        b-tensor (ensure-tensor b)
        a-shape (get-tensor-shape a-tensor)
        b-shape (get-tensor-shape b-tensor)]
    (cond
      (= a-shape b-shape) [a-tensor b-tensor]  ; Same shape
      (= (count a-shape) 1) [(tensor/->tensor (repeat (first b-shape) (first a-shape))) b-tensor]  ; Broadcast a
      (= (count b-shape) 1) [a-tensor (tensor/->tensor (repeat (first a-shape) (first b-shape)))]  ; Broadcast b
      :else (throw (ex-info "Incompatible shapes" {:a-shape a-shape :b-shape b-shape})))))

(defn train-network [network inputs targets epochs]
  "Train the network for the specified number of epochs"
  (loop [current-network network
         epoch 0]
    (if (>= epoch epochs)
      current-network
      (do
        (when (zero? (mod epoch 1))  ; Log every epoch instead of every 10
          (println "Training epoch:" epoch))
        (let [forward-results (doall (map #(forward-pass current-network %) inputs))  ; Force evaluation
              new-network (reduce (fn [net [forward-result target]]
                                  (backward-pass net forward-result target))
                                current-network
                                (map vector forward-results targets))]
          (recur new-network (inc epoch))))))
)
(defn create-game-network []
  "Create a neural network for the game:
   - Input layer: 49 neurons (7x7 board)
   - Hidden layer: 128 neurons with ReLU activation
   - Output layer: 1 neuron with sigmoid activation"
  (let [input-size 49
        hidden-size 128
        output-size 1]
    (create-network [input-size hidden-size output-size])))

(defn train-model [n-games]
  "Train a model on n games"
  (println "Starting model training on" n-games "games")
  (let [games (doall (repeatedly n-games #(f/play-game f/get-random-move)))  ; Force evaluation
        features (doall (mapv #(board->features (:board %)) games))  ; Force evaluation
        labels (doall (mapv #(tensor/->tensor [(if (f/solved? %) 1.0 0.0)]) games))  ; Force evaluation
        network (create-game-network)]
    (println "Created network with" (count (:layers network)) "layers")
    (println "Training on" (count features) "examples")
    (train-network network features labels 5)))  ; Reduced from 10 to 5 epochs for testing

(defn predict [game-state]
  "Use the model to predict the best move"
  (let [model (train-model 10)  ; Reduced from 100 to 10 games for testing
        features (board->features (:board game-state))
        forward-result (forward-pass model features)
        last-activation (last (:activations forward-result))
        prediction (double (first (flatten (seq last-activation))))  ; Extract numeric value from tensor
        moves (f/valid-moves (:board game-state))
        all-possible-moves (for [row (range 7)
                               col (range 7)
                               val (range 1 8)]
                           [row col val])
        temperature 1.0  ; Temperature for softmax scaling
        logits (map #(if (f/valid-move? (:board game-state) %)
                      (Math/exp (/ (rand) temperature))  ; Higher probability for valid moves
                      0.0)  ; Zero probability for invalid moves
                   all-possible-moves)
        probs (softmax logits)
        uniform-prob (/ 1.0 343)]  ; Equal probability for all possible moves
    (println "[DEBUG] NN prediction value:" prediction)
    (println "[DEBUG] Number of valid moves:" (count moves))
    (if (and (> prediction 0.5) (seq moves))
      {:policy (zipmap all-possible-moves probs)  ; Use temperature-scaled probabilities
       :value prediction}
      {:policy (zipmap all-possible-moves (repeat uniform-prob))  ; Even if prediction is low, return all moves
       :value prediction})))

(defn get-best-move [game-state]
  "Get the best move using the trained model"
  (when (nil? @trained-model)
    (throw (ex-info "Model not initialized" {})))
  (let [pipeline @trained-model
        result (run-pipeline pipeline game-state :transform)
        policy (:policy result)
        valid-moves (f/valid-moves (:board game-state))]
    (when (seq valid-moves)
      (apply max-key #(get policy % 0.0) valid-moves))))

(defn autoplay-from-position [game-state max-moves]
  "Autoplay from a given position using the neural network model"
  (println "\nStarting autoplay from position:")
  (let [board (:board game-state)]
    (println "Initial board:")
    (f/print-board board))
  (loop [state game-state
         moves-made 0
         moves []]
    (let [board (:board state)]
      (println "\nMove" moves-made ":")
      (println "Current board:")
      (f/print-board board)
      (if (f/game-over? state)
        (do
          (println "\nGame ended:")
          (println "Final board state:")
          (f/print-board board)
          (println "Moves made:" moves-made)
          (println "Game over?" (f/game-over? state))
          (println "Solved?" (f/solved? state))
          {:final-state state
           :moves-made moves-made
           :solved? (f/solved? state)
           :moves moves})
        (let [move (get-best-move state)]
          (if move
            (do
              (println "Making move:" move)
              (recur (f/make-move state move)
                     (inc moves-made)
                     (conj moves move)))
            (do
              (println "\nNo valid moves available:")
              (println "Final board state:")
              (f/print-board board)
              (println "Moves made:" moves-made)
              (println "Game over?" (f/game-over? state))
              (println "Solved?" (f/solved? state))
              {:final-state state
               :moves-made moves-made
               :solved? (f/solved? state)
               :moves moves})))))))

(defn analyze-position [game-state n-simulations]
  "Analyze a position by running multiple simulations.
   Returns a map with:
   - :win-rate - percentage of simulations that were solved
   - :avg-moves - average number of moves made
   - :best-move - most common first move
   - :move-stats - statistics for each possible first move"
  (let [simulations (repeatedly n-simulations #(autoplay-from-position game-state 100))
        solved-count (count (filter :solved? simulations))
        win-rate (/ (double solved-count) n-simulations)
        avg-moves (/ (double (reduce + (map :moves-made simulations))) n-simulations)
        first-moves (map #(first (:moves %)) simulations)
        move-stats (frequencies first-moves)
        best-move (key (apply max-key val move-stats))]
    {:win-rate win-rate
     :avg-moves avg-moves
     :best-move best-move
     :move-stats move-stats}))


;; Move selection functions
(defn get-policy-map [game-state]
  "Get policy map from neural network predictions"
  (let [predictions (predict game-state)
        policy (:policy predictions)
        moves (f/valid-moves (:board game-state))
        move-probs (zipmap moves (map #(double (get policy % 0.0)) moves))]  ; Ensure all probabilities are doubles
    (println "[DEBUG] Policy map size:" (count move-probs))
    (println "[DEBUG] Valid moves:" moves)
    move-probs))

(defn select-move [game-state]
  "Select a move based on neural network policy"
  (let [policy-map (get-policy-map game-state)
        moves (vec (keys policy-map))  ; Convert to vector for indexing
        probs (vec (map double (vals policy-map)))  ; Ensure all probabilities are doubles
        total-prob (reduce + probs)]
    (when (and (seq moves) (> total-prob 0.0))  ; Only proceed if we have moves and non-zero total probability
      (let [normalized-probs (map #(/ % total-prob) probs)
            selected-index (loop [r (rand)
                                remaining-probs (vec normalized-probs)  ; Convert to vector for indexing
                                index 0]
                           (if (or (empty? remaining-probs)
                                  (< r (first remaining-probs)))
                             index
                             (recur (- r (first remaining-probs))
                                   (vec (rest remaining-probs))  ; Convert to vector for indexing
                                   (inc index))))]
        (when (< selected-index (count moves))  ; Ensure index is valid
          (nth moves selected-index))))))

(defn initialize-model []
  "Initialize the neural network model if not already done"
  (when (nil? @trained-model)
    (println "Initializing neural network model...")
    (reset! trained-model (create-game-pipeline))))



