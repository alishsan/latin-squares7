(ns latin-squares7.nn
  (:require [latin-squares7.functions :as f]
            [latin-squares7.mcts :as mcts]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.tensor :as tensor]
            [tech.v3.datatype.functional :as df]))

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

(defn matrix-multiply [a b]
  "Matrix multiplication using tensor operations"
  (let [a-tensor (ensure-tensor a)
        b-tensor (ensure-tensor b)
        a-dims (get-tensor-dims a-tensor)
        b-dims (get-tensor-dims b-tensor)
        [a-rows a-cols] a-dims
        [b-rows b-cols] b-dims]
    (if (and (= a-cols b-rows) (> a-rows 0) (> b-cols 0))
      (tensor/->tensor
       (for [i (range a-rows)]
         (for [j (range b-cols)]
           (reduce + (map * (tensor-select a-tensor a-dims i)
                            (map #(nth % j) (seq b-tensor)))))))
      (throw (ex-info "Incompatible matrix dimensions for multiplication"
                     {:a-dims a-dims :b-dims b-dims})))))

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
  "Get the best move based on the model prediction"
  (let [{:keys [policy value]} (predict game-state)]
    (when (and (> value 0.5) (seq policy))
      (key (apply max-key val policy)))))

(defn autoplay-from-position [game-state max-moves]
  "Autoplay from a given position using the neural network model.
   Returns a map with:
   - :final-state - the final game state
   - :moves-made - number of moves made
   - :solved? - whether the game was solved
   - :moves - list of moves made"
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



