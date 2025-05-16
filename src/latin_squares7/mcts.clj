(ns latin-squares7.mcts
  (:require [functions :as f]
            [latin-squares7.nn :as nn]))

(defrecord Node [wins visits prior children])

(defn backpropagate [trie path result]
  (loop [t trie
         p path
         r result]
    (if (empty? p)
      t
      (let [move (peek p)
            node (get-in t (pop p) move)
            updated-node (-> node
                           (update :visits inc)
                           (update :wins + r))]
        (recur (assoc-in t (pop p) move updated-node)
               (pop p)
               (- r))))))

(defn select-node [trie path game-state]
  (let [current-node (get-in trie path)
        total-visits (reduce + (map :visits (vals current-node)))]
    (if (empty? (:children current-node))
      path
      (let [best-move (->> (:children current-node)
                           (map (fn [[move node]]
                                  [move (nn/ucb1 node total-visits)]))
                           (apply max-key second)
                           first)
            new-state (f/make-move game-state best-move)]
        (select-node trie (conj path best-move) new-state)))))

(defn expand-node [trie path game-state]
  (let [{:keys [policy]} (nn/predict game-state)
        legal-moves (f/suggested-moves (:board game-state))]
    (reduce (fn [t move]
              (assoc-in t (conj path move)
                        (->Node 0 0 (get policy move 0.1) {})))
            trie
            legal-moves)))

(defn simulate [game-state]
  (if (f/game-over? game-state)
    (if (= :alice (f/current-player game-state)) 1 -1)
    (let [{:keys [value]} (nn/predict game-state)]
      value)))

(defn mcts [initial-game-state iterations]
  (loop [trie {}
         i 0]
    (if (>= i iterations)
      trie
      (let [path (select-node trie [] initial-game-state)
            game-state (reduce f/make-move initial-game-state path)
            expanded-trie (expand-node trie path game-state)
            result (simulate game-state)
            new-trie (backpropagate expanded-trie path result)]
        (recur new-trie (inc i))))))

(defn best-move [trie game-state]
  (let [root-node (get-in trie [])
        moves-children (seq root-node)
        moves (map first moves-children)
        visits (map (comp :visits second) moves-children)
        max-visits (when (seq visits) (apply max visits))
        best-moves (when max-visits
                     (keep-indexed (fn [i v] (when (= v max-visits) (nth moves i)))
                         visits))]
    (when (seq best-moves)
      (rand-nth best-moves))))  ;; Random selection if multiple moves have same visits


