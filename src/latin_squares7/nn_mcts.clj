(ns latin-squares7.nn-mcts
  (:require [latin-squares7.functions :as f]
            [latin-squares7.nn-metamorph :as nn]
            [latin-squares7.mcts :as mcts]))

(defn get-neural-predictions [state]
  "Get policy and value predictions from neural network"
  (let [trained-pipeline (nn/train-model 10)  ; Train on 10 games first
        result (nn/run-pipeline trained-pipeline state :transform)
        policy (:policy result)
        value (:value result)]
    [policy value]))

(defn mcts-with-nn
  "Monte Carlo Tree Search with neural network policies"
  [state iterations]
  (let [[initial-policy initial-value] (get-neural-predictions state)
        initial-root (mcts/new-node state 1.0)  ;; Root node has prior 1.0
        tree (atom {:root initial-root})]
    
    ;; First expansion of root node
    (let [initial-moves (f/suggested-moves (:board state))
          initial-children (into {} (map (fn [move] 
                                         (let [move-key (mcts/compress-move move)
                                               prior (get initial-policy move-key 0.0)]
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
                    (let [[_ value] (get-neural-predictions (:state expanded-node))]
                      (or value 0.0)))  ;; Use 0.0 if value is nil
            _ (mcts/backpropagate expanded-path result)]))
    
    (mcts/best-move @tree)))

(defn auto-play-full-game []
  "Play a full game using MCTS with neural network policies"
  (loop [game-state (f/new-game)
         moves []]
    (if (f/game-over? game-state)
      {:board (:board game-state)
       :moves moves
       :solved? (f/solved? game-state)}
      (let [move (mcts-with-nn game-state 500)]  ;; Using 500 iterations for each move
        (if move
          (recur (f/make-move game-state move) (conj moves move))
          {:board (:board game-state)
           :moves moves
           :solved? (f/solved? game-state)})))))