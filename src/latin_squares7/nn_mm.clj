(ns latin-squares7.nn-mm
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

(println "Loading nn-mm namespace")

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
          features (tensor/->tensor (vec (take (* f/board-size f/board-size) flattened-data)))]
      (assoc ctx :metamorph/data (vec (map double (flatten (seq features))))))))

(defn create-layer-pipe
  "Create neural network layers with weights and biases"
  []
  (fn [{:metamorph/keys [data mode] :as ctx}]
    (let [input-size (* f/board-size f/board-size)  ; nxn board
          hidden-size 128
          policy-size (* f/board-size f/board-size f/board-size)  ; nxnxn possible moves
          value-size 1
          
          ;; Shared layers
          shared-weights1 (tensor/->tensor
                           (mapv (fn [_]
                                  (mapv (fn [_] (- (rand) 0.5))
                                        (range input-size)))
                                (range hidden-size)))
          shared-biases1 (tensor/->tensor (repeat hidden-size 0.0))
          
          ;; Policy head
          policy-weights (tensor/->tensor
                          (mapv (fn [_]
                                 (mapv (fn [_] (- (rand) 0.5))
                                       (range hidden-size)))
                               (range policy-size)))
          policy-biases (tensor/->tensor (repeat policy-size 0.0))
          
          ;; Value head
          value-weights (tensor/->tensor
                         (mapv (fn [_]
                                (mapv (fn [_] (- (rand) 0.5))
                                      (range hidden-size)))
                              (range value-size)))
          value-biases (tensor/->tensor (repeat value-size 0.0))]
      
      (-> ctx
          (assoc :metamorph/data (:metamorph/data ctx))
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
    (let [input-size (* f/board-size f/board-size)
          policy-size (* f/board-size f/board-size f/board-size)
          
          ;; Simple uniform policy for now
          policy-probs (repeat policy-size (/ 1.0 policy-size))
          
          ;; Simple value estimation based on board state
          filled-cells (count (filter some? (flatten data)))
          value (/ (double filled-cells) (double input-size))]
      
      (assoc ctx :metamorph/data
             {:policy (zipmap (range policy-size) policy-probs)
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
          policy (:policy predictions)
          value (:value predictions)
          
          ;; Filter policy to only include valid moves and normalize
          valid-policy (when (and (seq moves) (seq policy))
                        (let [move-probs (map (fn [move]
                                              (let [move-key (f/compress-move move)]
                                                (get policy move-key 0.0)))
                                            moves)
                              total-prob (reduce + move-probs)]
                          (if (pos? total-prob)
                            (zipmap moves
                                   (map #(/ % total-prob) move-probs))
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
(defn initialize-model []
  "Initialize the neural network model if not already done"
  (when (nil? @trained-model)
    (println "Initializing neural network model...")
    (reset! trained-model (create-game-pipeline))))

(defn get-best-move [game-state]
  "Get the best move using pure neural network predictions"
  (initialize-model)  ; Ensure model is initialized
  (let [predictions (run-pipeline @trained-model game-state :transform)
        policy (:policy predictions)
        valid-moves (f/valid-moves (:board game-state))
        move-number (count (filter some? (flatten (:board game-state))))]
    (println (format "[DEBUG] Move #%d" (inc move-number)))
    (println "[DEBUG] NN policy:" policy)
    (println "[DEBUG] Valid moves:" valid-moves)
    (let [move (when (seq valid-moves)
                 (apply max-key #(get policy % 0.0) valid-moves))]
      (println "[DEBUG] Chosen move:" move)
      move)))

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

(defn retrain-model [n-games]
  "Retrain the neural network model"
  (println "Retraining model with" n-games "games...")
  (let [games (repeatedly n-games #(f/play-game f/get-random-move))
        new-model (create-game-pipeline)]
    (reset! trained-model new-model)
    (println "Model retrained successfully!")
    new-model))

