(ns latin-squares7.mcts
  (:require [functions :as f]
            [clojure.math :as math]))

;; Node representation
(defrecord Node [wins visits prior children])

(defn new-node
  "Create a new MCTS node with default values"
  ([] (->Node 0.0 0 0.1 {}))
  ([prior] (->Node 0.0 0 prior {})))

;; UCB1 formula for move selection
(defn ucb1 [node total-visits]
  (let [wins (:wins node 0.0)
        visits (:visits node 0)
        prior (:prior node 0.1)]
    (if (zero? visits)
      Double/POSITIVE_INFINITY
      (+ (/ wins visits) 
         (* prior (math/sqrt (/ (math/log (inc total-visits)) visits)))))))

(defn select-path [trie game-state]
  "Select a path through the tree using UCB1"
  (loop [path []
         node (get trie [] (new-node))
         state game-state]
    (if (or (empty? (:children node))
            (f/game-over? state))
      path
      (let [total-visits (reduce + (map :visits (vals (:children node))))
            [best-move child] (->> (:children node)
                               (map (fn [[m n]] [m (ucb1 n total-visits)]))
                               (apply max-key second))
            new-state (f/make-move state best-move)]
        (if new-state
          (recur (conj path best-move) child new-state)
          path)))))

(defn expand-node [trie path game-state]
  "Expand the tree by adding legal moves and update parent's :children map"
  (let [legal-moves (f/suggested-moves (:board game-state))]
    (if (seq legal-moves)
      (let [moves (take 50 (shuffle legal-moves))
            trie-with-children (reduce (fn [t move]
                                         (assoc-in t (conj path move) (new-node)))
                                       trie
                                       moves)
            ;; Use the actual child nodes from the trie
            children-map (into {} (map (fn [move] [move (get-in trie-with-children (conj path move))]) moves))]
        (update-in trie-with-children path #(assoc % :children children-map)))
      trie)))

(defn backpropagate [trie path result]
  "Backpropagate results through the tree and update parent :children maps"
  (let [result (double result)]
    (loop [t trie
           p []
           [m & more] path
           r result
           perspective 1.0]
      (if (nil? m)
        t
        (let [node (get-in t (conj p m) (new-node))
              updated (-> node
                        (update :wins + (* r perspective))
                        (update :visits inc))
              t' (assoc-in t (conj p m) updated)
              ;; Always update parent's :children map, even for root
              t'' (update-in t' p #(assoc % :children (assoc (:children %) m updated)))]
          (recur t'' (conj p m) more (- r) (- perspective)))))))

(defn simulate [game-state]
  "Run a random simulation from current state"
  (loop [state game-state
         depth 0
         max-depth 50
         current-player (f/current-player state)]
    (cond
      (f/game-over? state) (if (= current-player :alice) -1.0 1.0)
      (>= depth max-depth) 0.0
      :else (let [moves (f/suggested-moves (:board state))]
              (if (empty? moves)
                (if (= current-player :alice) -1.0 1.0)
                (let [move (rand-nth moves)
                      new-state (f/make-move state move)]
                  (recur new-state (inc depth) max-depth 
                         (if (= current-player :alice) :bob :alice))))))))

(defn mcts [initial-game-state iterations]
  "Main MCTS function"
  (loop [trie {[] (new-node)} ; Initialize with root node
         i 0]
    (let [path (select-path trie initial-game-state)
          game-state (reduce (fn [gs m] (or (f/make-move gs m) gs))
                            initial-game-state
                            path)
          node (get-in trie path)
          need-expand (and (not (f/game-over? game-state))
                            (empty? (:children node)))
          [trie path game-state skip?] (if need-expand
                                          (let [t (expand-node trie path game-state)
                                                moves (keys (:children (get-in t path)))]
                                            (if (seq moves)
                                              [(assoc t [] (get-in t []))
                                               (conj path (first moves))
                                               (f/make-move game-state (first moves))
                                               false]
                                              [(assoc t [] (get-in t []))
                                               path
                                               game-state
                                               true]))
                                          [(assoc trie [] (get-in trie []))
                                           path
                                           game-state
                                           false])
          result (when-not skip? (simulate game-state))
          updated-trie (if-not skip?
                         (backpropagate trie path result)
                         trie)
          updated-trie (assoc updated-trie [] (get-in updated-trie []))]
      (when (zero? (mod i 100))
        (let [root (get updated-trie [])
              last-move (last path)]
          (println "Iteration" i 
                   "| Nodes:" (count (keys updated-trie))
                   "| Last move:" last-move
                   "| Root wins:" (when root (:wins root))
                   "| Root visits:" (when root (:visits root)))))
      (if (>= i iterations)
        updated-trie
        (recur updated-trie (inc i))))))

(defn best-move
  "Select the best move from a given trie and node path.
   Args:
   - trie: The MCTS search tree
   - path: The path to the node in the trie (default: root [])"
  ([trie]
   (best-move trie []))
  ([trie path]
   (let [node (get-in trie path)
         children (:children node)]
     (when (seq children)
       (let [[best-move _] (->> children
                             (sort-by (fn [[_ {:keys [visits wins]}]]
                                      (/ wins (max 1 visits)))
                                    >)
                             first)]
         best-move)))))

(defn print-node-stats [node]
  (println "Wins:" (:wins node) "| Visits:" (:visits node)
  (println "Children:" (count (:children node)))))

(defn inspect-trie [trie path]
  (let [node (get-in trie path)]
    (println "Node at path" path ":")
    (print-node-stats node)
    (when-let [children (:children node)]
      (println "Top 3 children:")
      (doseq [[move child] (take 3 (sort-by (comp :visits second) > children))]
        (println "Move:" move "-> Wins:" (:wins child) "Visits:" (:visits child))))))
