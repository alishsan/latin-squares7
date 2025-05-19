(ns latin-squares7.mcts
  (:require [functions :as f]
            [latin-squares7.nn :as nn]
            [clojure.math :as math]))

;; Optimized node representation
(defrecord Node [^double wins 
                 ^long visits 
                 ^double prior 
                 ^clojure.lang.PersistentTreeMap children])

(defn new-node
  ([] (->Node 0.0 0 0.1 (sorted-map)))
  ([prior] (->Node 0.0 0 prior (sorted-map))))

;; Move compression (3 numbers -> 1 long)
(defn compress-move [[r c n]]
  (bit-or (bit-shift-left r 16)
          (bit-shift-left c 8)
          n))


;; UCB1 formula for move selection
(defn ucb1 [node total-visits]
  (let [wins (:wins node 0)
        visits (:visits node 0)
        prior (:prior node 1.0)]
    (if (zero? visits)
      Double/POSITIVE_INFINITY  ; Always explore unvisited nodes
      (+ (/ wins visits) 
         (* prior (Math/sqrt (/ (Math/log total-visits) visits)))))))
         
(defn decompress-move [m]
  [(bit-shift-right m 16)
   (bit-and 0xFF (bit-shift-right m 8))
   (bit-and 0xFF m)])

;; Memory management
(def MAX-NODES 100000)
(def CLEANUP-RATIO 0.5)

(defn node-count [trie]
  (count trie))

(defn prune-trie [trie]
  (if (> (node-count trie) MAX-NODES)
    (let [threshold (->> trie
                      vals
                      (map :visits)
                      (sort)
                      (nth (int (* (count trie) CLEANUP-RATIO))))]
      (into {} (remove #(< (:visits (val %)) threshold) trie))
    trie)))

(defn select-path [trie game-state]
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
            new-state (f/make-move state (decompress-move best-move))]
        (if new-state ; Only continue if move was valid
          (recur (conj path best-move)
                 child
                 new-state)
          path)))))



(defn expand-node [trie path game-state]
  (let [legal-moves (->> (f/suggested-moves (:board game-state))
                         (map compress-move))]
    (if (seq legal-moves)
      (reduce (fn [t move]
                (assoc-in t (conj path move) (new-node (/ 1.0 (count legal-moves)))))
              trie
              (take 50 (shuffle legal-moves))) ; Limit and shuffle for variety
      trie)))

(defn backpropagate [trie path result]
  (let [result (double result)
        perspective (if (even? (count path)) 1.0 -1.0)] ; Track player perspective
    (loop [t trie
           [m & more] path
           r result
           pers perspective]
      (if (nil? m)
        t
        (let [node (get t m (new-node))
              updated (-> node
                       (update :wins + (* r pers))
                       (update :visits inc))]
          (recur (assoc t m updated)
                 more
                 (- r)
                 (- pers))))))) ; Alternate perspective


(defn simulate [game-state]
  (loop [state game-state
         depth 0
         max-depth 50
         current-player (f/current-player state)]
    (cond
      (f/game-over? state)
      (if (= current-player :alice) -1.0 1.0)
      
      (>= depth max-depth)
      0.0
      
      :else
      (let [moves (f/suggested-moves (:board state))]
        (if (empty? moves)
          (if (= current-player :alice) -1.0 1.0)
          (let [move (rand-nth moves)
                new-state (f/make-move state move)]
            (recur new-state
                   (inc depth)
                   max-depth
                   (if (= current-player :alice) :bob :alice))))))))




(defn mcts [initial-game-state iterations]
  (loop [trie {[] (new-node)}
         i 0]
    (let [path (select-path trie initial-game-state)
          game-state (reduce (fn [gs m] 
                              (or (f/make-move gs (decompress-move m)) 
                              gs))
                            initial-game-state
                            path)
          trie (if (and (not (f/game-over? game-state))
                        (empty? (get-in trie path)))
                (expand-node trie path game-state)
                trie)
          result (simulate game-state)
          updated-trie (backpropagate trie path result)]
      
      (when (zero? (mod i 100))
        (let [root (get updated-trie [])
              last-move (when (seq path) (decompress-move (last path)))]
          (println "Iteration" i 
                   "| Nodes:" (node-count updated-trie)
                   "| Last move:" last-move
                   "| Root wins:" (:wins root)
                   "| Root visits:" (:visits root)))
      
      (if (>= i iterations)
        updated-trie
        (recur updated-trie (inc i)))))))

(defn best-move [trie]
  (when-let [root (get trie [])]
    (when-let [children (seq (:children root))]
      (let [[best-move _] (->> children
                            (sort-by (fn [[_ {:keys [visits wins]}]] 
                                      (/ wins (max 1 visits)))
                                    >)
                            first)]
        (decompress-move best-move)))))





