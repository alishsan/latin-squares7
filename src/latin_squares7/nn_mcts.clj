(ns latin-squares7.nn-mcts
  (:require [latin-squares7.mcts :as mcts]
            [latin-squares7.nn-metamorph :as nn]
            [latin-squares7.functions :as f]))

(defn get-best-move
  "Get the best move using neural network guided MCTS"
  [state]
  (let [model (nn/get-trained-model)
        predictions (if model
                     (nn/run-pipeline model state :transform)
                     {:policy {} :value 0.0})
        policy (:policy predictions)
        value (:value predictions)
        
        ;; Use neural network policy as priors for MCTS
        move (mcts/mcts state 500 policy)]  ; Pass the policy to MCTS
    move))

(defn autoplay-from-position
  "Autoplay from a given position using neural network guided MCTS"
  [state max-moves]
  (f/autoplay-from-position state max-moves get-best-move)) 