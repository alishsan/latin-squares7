(ns latin-squares7.nn
  (:require [latin-squares7.functions :as f]
            [latin-squares7.mcts :as mcts]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.tensor :as tensor]
            [scicloj.ml.core :as ml]
            [scicloj.ml.smile.classification :as smile]))

(defn board->features [board]
  "Convert a 7x7 board into a feature vector for the neural network"
  (let [flattened (flatten board)]
    (mapv #(if (nil? %) 0 %) flattened)))

(defn create-training-data [num-games]
  "Generate training data by simulating games"
  (let [games (repeatedly num-games mcts/auto-play-full-game)
        features (map #(board->features (:board %)) games)
        labels (map #(if (f/solved? %) 1 0) games)]
    (-> (ds/->dataset {:features features
                       :labels labels})
        (ds-mod/prepare-for-model))))

(defn train-model [n-games]
  "Train a model on n games"
  (let [data (create-training-data n-games)
        model (ml/train data
                       {:model-type :gradient-boost
                        :max-iterations 100
                        :max-depth 5})]
    model))

(defn predict [game-state]
  "Use the model to predict the best move"
  (let [model (train-model 100)  ; Train on 100 games
        features (board->features (:board game-state))
        prediction (ml/predict model features)
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



