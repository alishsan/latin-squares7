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

(defn get-tensor-dims [tensor]
  "Get tensor dimensions, handling both tensor and scalar cases"
  (if (number? tensor)
    [1 1]  ; For scalar values, treat as 1x1 tensor
    (let [data (seq tensor)]
      (if (nil? data)
        [1 1]  ; For nil or empty tensors, treat as 1x1
        (let [rows (count data)
              first-row (first data)]
          (if (number? first-row)
            [rows 1]  ; For 1D tensors, treat as Nx1
            [rows (count first-row)]))))))  ; For 2D tensors, get actual dimensions

(defn tensor-select [tensor dims idx]
  "Safely select from a tensor based on its dimensions"
  (let [[rows cols] dims]
    (cond
      (number? tensor) tensor  ; If it's a scalar, return as is
      (= rows 1) (if (= cols 1)
                   tensor  ; 1x1 tensor
                   (tensor/select tensor 0 idx))  ; 1xN tensor
      (= cols 1) (tensor/select tensor idx 0)  ; Nx1 tensor
      :else (tensor/select tensor idx :all))))  ; NxM tensor

(defn matrix-multiply [a b]
  "Matrix multiplication using basic tensor operations"
  (let [a-dims (get-tensor-dims a)
        b-dims (get-tensor-dims b)
        a-rows (first a-dims)
        b-cols (second b-dims)
        result (tensor/->tensor (repeat a-rows (repeat b-cols 0.0)))]
    (loop [i 0]
      (if (>= i a-rows)
        result
        (let [row (tensor-select a a-dims i)
              new-row (loop [j 0
                           acc []]
                       (if (>= j b-cols)
                         (tensor/->tensor acc)
                         (let [col (tensor-select b b-dims j)
                               dot-product (df/+ (df/* row col))]
                           (recur (inc j) (conj acc dot-product)))))]
          (recur (inc i)))))))

(defn transpose [tensor]
  "Transpose a tensor using the correct dimensions"
  (let [dims (get-tensor-dims tensor)
        [rows cols] dims]
    (cond
      (number? tensor) tensor  ; Scalar remains unchanged
      (= rows 1) (if (= cols 1)
                   tensor  ; 1x1 tensor remains unchanged
                   (tensor/->tensor (map vector (seq tensor))))  ; Convert 1xN to Nx1
      (= cols 1) (tensor/->tensor [(seq tensor)])  ; Convert Nx1 to 1xN
      :else (let [result (tensor/->tensor (repeat cols (repeat rows 0.0)))]
              (loop [i 0]
                (if (>= i rows)
                  result
                  (let [row (tensor-select tensor dims i)]
                    (loop [j 0]
                      (if (>= j cols)
                        (recur (inc i))
                        (do
                          (tensor/mset! result j i (tensor/mget row j))
                          (recur (inc j))))))))))))

(defn forward-pass [network input]
  "Perform a forward pass through the network"
  (loop [layers (:layers network)
         activations [input]
         zs []]
    (if (empty? layers)
      {:activations (vec activations)
       :zs (vec zs)}
      (let [layer (first layers)
            z (df/+ (matrix-multiply (:weights layer) (last activations))
                   (:biases layer))
            activation (tensor/typed-compute-tensor :float64 [1] [z] sigmoid)]
        (recur (rest layers)
               (conj activations activation)
               (conj zs z))))))

(defn backward-pass [network forward-result target]
  "Perform backpropagation to update weights"
  (let [activations (:activations forward-result)
        zs (:zs forward-result)
        layers (:layers network)
        learning-rate (:learning-rate network)
        
        ;; Calculate output layer error
        output-activation (last activations)
        output-error (df/- output-activation target)
        output-delta (df/* output-error
                          (tensor/typed-compute-tensor :float64 [1] [output-activation] sigmoid-derivative))
        
        ;; Backpropagate error
        deltas (reduce (fn [deltas [layer z activation prev-activation]]
                        (let [weight-delta (matrix-multiply (first deltas)
                                                          (transpose prev-activation))
                              bias-delta (first deltas)
                              error (matrix-multiply (transpose (:weights layer))
                                                   (first deltas))
                              delta (df/* error
                                        (tensor/typed-compute-tensor :float64 [1] [activation] sigmoid-derivative))]
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
                                                 (matrix-multiply delta
                                                                (transpose prev-activation)))
                               bias-update (df/* learning-rate delta)]
                           {:weights (df/- (:weights layer) weight-update)
                            :biases (df/- (:biases layer) bias-update)}))
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



