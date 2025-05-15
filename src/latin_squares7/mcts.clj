(ns latin-squares7.mcts
  (:require [functions :as f]
            [clojure.set :as set]
            [clojure.string :as str]
            [latin-squares7.nn :as nn])) ; Add NN namespace

;; Enhanced Node structure with neural net priors
(defrecord Node [prior wins visits children])

;; Neural Network enhanced selection
(defn select-node [network trie current-path game-state]
  (let [valid-moves (f/suggested-moves (:board game-state))
        total-visits (reduce + (map :visits (vals (get-in trie current-path))))]
    (if (empty? valid-moves)
      current-path
      (let [best-move (->> valid-moves
                           (map (fn [move]
                                  (let [path (conj current-path move)
                                        node (get-in trie path)]
                                    [move (nn/ucb1 node prior total-visits)])))
                           (apply max-key second)
                           first)
            new-state (f/make-move game-state best-move)]
        (if (zero? (get-in trie [(conj current-path best-move) :visits] 0))
          (conj current-path best-move)
          (select-node network trie (conj current-path best-move) new-state))))))

;; Neural Network enhanced expansion
(defn expand-node [network trie path game-state]
  (let [[policy value] (nn/predict (f/board->tensor (:board game-state) (f/current-player game-state)))
        legal-moves (f/suggested-moves (:board game-state))]
    (reduce (fn [t move]
              (assoc-in t (conj path move)
                        (->Node (get policy move) 0 0 {})))
            trie
            legal-moves)))

;; Simulation with neural network value estimation
(defn simulate [network game-state]
  (if (f/game-over? game-state)
    (if (= (f/current-player game-state) :alice) 1 -1) ; [-1, 1] range
    (let [[_ value] (nn/predict (f/board->tensor (:board game-state) (f/current-player game-state)))]
      value)))

;; Enhanced MCTS main loop
(defn mcts [network initial-game-state iterations]
  (let [root-path []]
    (loop [trie {}
           i 0]
      (if (>= i iterations)
        trie
        (let [path (select-node network trie root-path initial-game-state)
              game-state (reduce f/make-move initial-game-state path)
              expanded-trie (expand-node network trie path game-state)
              result (simulate network game-state)
              new-trie (backpropagate expanded-trie path result)]
          (recur new-trie (inc i)))))))

;; Neural Network training data generation
(defn generate-training-data [network game-state iterations]
  (let [trie (mcts network game-state iterations)
        total-visits (reduce + (map :visits (vals (get-in trie []))))
        policy (into {}
                     (for [[move node] (get-in trie [])]
                       [move (/ (:visits node) total-visits)]))
        value (:wins (node-stats trie []))]
    {:board (f/board->tensor (:board game-state) (f/current-player game-state))
     :policy policy
     :value value}))

;; Keep your existing parallel-mcts and best-move functions
;; Add neural network based best move selection
(defn nn-best-move [network game-state iterations]
  (let [trie (mcts network game-state iterations)
        moves (f/suggested-moves (:board game-state))
        visit-counts (map #(get-in trie [% :visits] 0) moves)]
    (nth moves (.indexOf visit-counts (apply max visit-counts)))))
