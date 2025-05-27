(ns latin-squares7.nn
  (:require [latin-squares7.functions :as f]
            [latin-squares7.mcts :as mcts]
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

;; Neural Network Implementation
(defn sigmoid [x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn sigmoid-derivative [x]
  (* x (- 1.0 x)))

(defn relu [x]
  (max 0.0 x))

(defn relu-derivative [x]
  (if (> x 0.0) 1.0 0.0))

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
                  (create-layer (if (= i 0)
                                 (first layer-sizes)
                                 (nth layer-sizes (dec i)))
                               size))
                (rest layer-sizes))]
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

(defn tensor-add [a b]
  "Add tensors with broadcasting using df/+"
  (df/+ (ensure-tensor a) (ensure-tensor b)))

(defn tensor-multiply [a b]
  "Multiply tensors with broadcasting using df/*"
  (df/* (ensure-tensor a) (ensure-tensor b)))

(defn transpose [tensor]
  "Transpose a tensor using df/transpose"
  (df/transpose (ensure-tensor tensor)))

(defn forward-pass [network input]
  "Perform a forward pass through the network"
  (loop [layers (:layers network)
         activations [(ensure-tensor input)]
         zs []]
    (if (empty? layers)
      {:activations (vec activations)
       :zs (vec zs)}
      (let [layer (first layers)
            weights (ensure-tensor (:weights layer))
            biases (ensure-tensor (:biases layer))
            last-activation (ensure-tensor (last activations))
            z (df/+ (df/matmul weights last-activation)
                   biases)
            activation (df/map sigmoid z)]
        (recur (rest layers)
               (conj activations activation)
               (conj zs z))))))

(defn backward-pass [network forward-result target]
  "Perform backpropagation to update weights"
  (let [activations (:activations forward-result)
        zs (:zs forward-result)
        layers (:layers network)
        learning-rate (to-double (:learning-rate network))
        
        ;; Calculate output layer error
        output-activation (last activations)
        target-tensor (ensure-tensor target)
        output-error (df/- output-activation target-tensor)
        output-delta (df/* output-error
                          (df/map sigmoid-derivative output-activation))
        
        ;; Backpropagate error
        deltas (reduce (fn [deltas [layer z activation prev-activation]]
                        (let [weight-delta (df/matmul (first deltas)
                                                    (df/transpose prev-activation))
                              bias-delta (first deltas)
                              error (df/matmul (df/transpose (:weights layer))
                                             (first deltas))
                              delta (df/* error
                                        (df/map sigmoid-derivative activation))]
                          (conj deltas delta)))
                      [output-delta]
                      (map vector
                           (reverse (rest layers))
                           (reverse (rest zs))
                           (reverse (rest activations))
                           (reverse (butlast activations))))
        
        ;; Update weights and biases
        new-layers (map (fn [layer delta activation prev-activation]
                         (let [weight-update (df/* learning-rate 
                                                 (df/matmul delta
                                                           (df/transpose prev-activation)))
                               bias-update (df/* learning-rate delta)]
                           {:weights (df/- (:weights layer) weight-update)
                            :biases (df/+ (:biases layer) bias-update)}))
                       (reverse layers)
                       (reverse deltas)
                       (reverse (rest activations))
                       (reverse (butlast activations)))]
    (assoc network :layers (vec (reverse new-layers)))))

(defn train-network [network inputs targets epochs]
  "Train the network for the specified number of epochs"
  (loop [current-network network
         epoch 0]
    (if (>= epoch epochs)
      current-network
      (let [forward-results (map #(forward-pass current-network %) inputs)
            new-network (reduce (fn [net [forward-result target]]
                                (backward-pass net forward-result target))
                              current-network
                              (map vector forward-results targets))]
        (recur new-network (inc epoch))))))

;; Game-specific functions
(defn board->features [board]
  "Convert a 7x7 board into a feature vector for the neural network"
  (let [flattened (flatten board)]
    (tensor/->tensor
     (mapv #(if (nil? %) 0.0 %) flattened))))

(defn create-game-network []
  "Create a neural network for the game:
   - Input layer: 49 neurons (7x7 board)
   - Hidden layer: 128 neurons with ReLU activation
   - Output layer: 1 neuron with sigmoid activation"
  (create-network [49 128 1]))

(defn train-model [n-games]
  "Train a model on n games"
  (let [games (repeatedly n-games mcts/auto-play-full-game)
        features (mapv #(board->features (:board %)) games)
        labels (mapv #(tensor/->tensor [(if (f/solved? %) 1.0 0.0)]) games)
        network (create-game-network)]
    (train-network network features labels 100)))  ; 100 epochs

(defn predict [game-state]
  "Use the model to predict the best move"
  (let [model (train-model 100)  ; Train on 100 games
        features (board->features (:board game-state))
        forward-result (forward-pass model features)
        prediction (first (last (:activations forward-result)))
        moves (f/suggested-moves (:board game-state))]
    (if (and (> prediction 0.5) (seq moves))
      {:policy (zipmap moves (repeat (/ 1.0 (count moves))))  ; Uniform priors
       :value prediction}
      {:policy {}  ; Empty policy if no confident prediction
       :value 0.0})))

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

;; MCTS with Neural Network
(defn mcts-with-nn
  "Monte Carlo Tree Search with neural network policies"
  [state iterations]
  ;; Initialize model if needed
  (when (nil? @trained-model)
    (println "Initializing neural network model...")
    (set-trained-model (train-on-self-play 100)))
    
  (let [initial-root (mcts/new-node state 1.0)  ;; Root node has prior 1.0
        tree (atom {:root initial-root})]
    
    ;; First expansion of root node with neural network priors
    (let [initial-moves (f/suggested-moves (:board state))
          initial-children (into {} (map (fn [move] 
                                         (let [move-key (mcts/compress-move move)
                                               prior (get-in (run-pipeline @trained-model state :transform)
                                                           [:policy move] 0.0)]
                                           [move-key 
                                            (mcts/new-node (f/make-move state move) prior)]))
                                       initial-moves))
          root-with-children (assoc initial-root :children initial-children)]
      (reset! tree {:root root-with-children}))
    
    (dotimes [_ iterations]
      (let [path (mcts/select-path @tree)
            node (get-in @tree (cons :root path))
            expanded-path (if (and (empty? (:children node))
                                 (not (f/game-over? (:state node))))
                          (mcts/expand-node node)
                          path)
            expanded-node (get-in @tree (cons :root expanded-path))
            result (if (f/game-over? (:state expanded-node))
                    (if (f/solved? (:state expanded-node)) 1.0 0.0)
                    (get-in (run-pipeline @trained-model (:state expanded-node) :transform)
                           [:value] 0.0))  ;; Use neural network value
            _ (mcts/backpropagate expanded-path result)]))
    
    (mcts/best-move @tree)))

(defn self-play-game []
  "Play a full game using neural-guided MCTS and return the game history"
  (loop [game-state (f/new-game)
         moves []
         history []]
    (if (f/game-over? game-state)
      {:board (:board game-state)
       :moves moves
       :solved? (f/solved? game-state)
       :history history}
      (let [move (mcts-with-nn game-state 500)]  ;; Using 500 iterations for each move
        (if move
          (recur (f/make-move game-state move)
                 (conj moves move)
                 (conj history {:state game-state
                              :move move
                              :result (if (f/solved? (f/make-move game-state move))
                                       1.0
                                       0.0)}))
          {:board (:board game-state)
           :moves moves
           :solved? (f/solved? game-state)
           :history history})))))

(defn train-on-self-play [n-games]
  "Train the model through self-play games"
  (let [pipeline (create-game-pipeline)
        games (repeatedly n-games self-play-game)
        trained-model (reduce (fn [model game]
                              (try
                                (println "\n=== Training on Self-Play Game ===")
                                (println "Game result:" (if (:solved? game) "Solved" "Not Solved"))
                                
                                ;; Train on each position in the game
                                (reduce (fn [current-model {:keys [state move result]}]
                                        (let [result (run-pipeline current-model state :fit)]
                                          (if (contains? result :layers)
                                            result
                                            current-model)))
                                      model
                                      (:history game))
                                
                                (catch Exception e
                                  (println "\nError during training:")
                                  (println "Exception message:" (.getMessage e))
                                  model)))  ; Return unchanged model on error
                            pipeline
                            games)]
    trained-model))

(defn retrain-model [num-games]
  "Retrain the model on new self-play games"
  (println "Retraining model on" num-games "self-play games...")
  (set-trained-model (train-on-self-play num-games)))



