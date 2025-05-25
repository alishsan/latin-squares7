(ns latin-squares7.mcts
  (:require [functions :as f]
            [clojure.math :as math]
            [clojure.java.io :as io]))

;; Debug logging setup
(def debug-log-file "mcts-debug.log")

(defn debug-log [& args]
  (with-open [w (io/writer debug-log-file :append true)]
    (.write w (str (apply str args) "\n"))))

;; Node representation
(defrecord Node [wins visits prior children])

;; Custom print-method for Node to make output more concise
(defmethod print-method Node [node writer]
  (.write writer (format "Node{w:%.1f v:%d p:%.1f c:%d}"
                        (:wins node)
                        (:visits node)
                        (:prior node)
                        (count (:children node)))))

(defn new-node
  "Create a new MCTS node with default values"
  ([] (->Node 0.0 0 0.1 {}))
  ([prior] (->Node 0.0 0 prior {})))

;; UCB1 formula for move selection
(defn ucb1 [node total-visits]
  (let [wins (:wins node 0.0)
        visits (:visits node 0)
        prior (:prior node 0.1)
        exploration-constant 1.5]  ;; Balanced between exploration and exploitation
    (if (zero? visits)
      (+ Double/POSITIVE_INFINITY (* 0.1 (rand)))  ;; Small randomization for unvisited nodes
      (+ (/ wins (max 1 visits))  ;; Exploitation term
         (* exploration-constant 
            (math/sqrt (/ (math/log (inc total-visits)) (max 1 visits))))  ;; Exploration term
         (* prior 0.1)  ;; Prior knowledge term
         (* 0.05 (rand))))))  ;; Small noise for diversity

;; Utility functions for move compression
(defn compress-move [[r c n]]
  (+ (* 100 r) (* 10 c) n))

(defn decompress-move [move-int]
  [(quot move-int 100)
   (quot (mod move-int 100) 10)
   (mod move-int 10)])

;; Update select-path to use compressed moves
(defn select-path [tree game-state]
  "Select a path through the tree using UCB1"
  (loop [path []
         node (:root tree)
         state game-state
         depth 0]
    (if (or (empty? (:children node))
            (f/game-over? state)
            (nil? state)
            (>= depth 10))  ;; Limit search depth
      path
      (let [total-visits (reduce + (map :visits (vals (:children node))))
            moves (->> (:children node)
                    (map (fn [[m n]] [m (ucb1 n total-visits)]))
                    (sort-by second >))
            temperature 0.5
            move-probs (->> moves
                          (map (fn [[m score]] [m (math/exp (/ score temperature))]))
                          (sort-by second >))
            total-prob (reduce + (map second move-probs))
            normalized-probs (map (fn [[m p]] [m (/ p total-prob)]) move-probs)
            r (rand)
            [best-move _] (loop [[[m p] & more] normalized-probs
                               cum-prob 0.0]
                           (if (or (nil? m) (>= cum-prob r))
                             [m p]
                             (recur more (+ cum-prob p))))
            move-vec (when best-move (decompress-move best-move))
            new-state (when (and state move-vec (f/valid-move? (:board state) move-vec))
                       (f/make-move state move-vec))]
        (if new-state
          (recur (conj path best-move)
                 (get (:children node) best-move)
                 new-state
                 (inc depth))
          path)))))

;; Update expand-node to use compressed moves as keys
(defn expand-node [tree path game-state]
  "Expand the tree by adding legal moves and update parent's :children map"
  (let [legal-moves (f/suggested-moves (:board game-state))
        _ (println "[expand-node] legal-moves:" legal-moves)]
    (if (seq legal-moves)
      (let [limited-moves (take 50 legal-moves)  ;; Limit to 50 moves first
            _ (println "[expand-node] limited-moves:" limited-moves)
            children-map (reduce (fn [acc move]
                                 (let [compressed (compress-move move)]
                                   (assoc acc compressed (new-node))))
                               {}
                               limited-moves)
            _ (println "[expand-node] children-map keys:" (keys children-map))
            parent-node (get-in tree (cons :root path) (new-node))
            updated-parent (assoc parent-node :children children-map)
            _ (println "[expand-node] updated-parent children count:" (count (:children updated-parent)))
            updated-tree (assoc-in tree (cons :root path) updated-parent)
            _ (println "[expand-node] Final tree root children count:" (count (:children (:root updated-tree))))]
        (if (empty? path)
          (do
            (println "[expand-node] Updating root node")
            (let [root-node (:root updated-tree)
                  updated-root (assoc root-node :children children-map)
                  final-tree (assoc updated-tree :root updated-root)]
              (println "[expand-node] Final root children count:" (count (:children (:root final-tree))))
              final-tree))
          (do
            (println "[expand-node] Updating non-root node")
            (let [root-node (:root updated-tree)
                  final-tree (assoc updated-tree :root root-node)]
              (println "[expand-node] Final root children count:" (count (:children (:root final-tree))))
              final-tree))))
      (do 
        (println "[expand-node] No legal moves, creating default children")
        (let [default-move [0 0 1]
              default-compressed (compress-move default-move)
              default-node (new-node)
              default-children {default-compressed default-node}
              root-node (-> (new-node)
                          (assoc :children default-children)
                          (assoc :visits 1))]
          (println "[expand-node] Created default children for root node")
          {:root root-node})))))

;; Update backpropagate to use compressed moves in path
(defn backpropagate [tree path result]
  "Backpropagate results through the tree and update node statistics."
  (let [result (double result)]
    (let [t-final (loop [t tree
                         p []
                         [m & more] path
                         r result
                         perspective 1.0]
                    (if (nil? m)
                      t
                      (let [current-path (cons :root (conj p m))
                            node (get-in t current-path (new-node))
                            player-result (if (= perspective 1.0)
                                          (if (pos? r) 1.0 0.0)
                                          (if (neg? r) 1.0 0.0))
                            updated-node (-> node
                                           (update :wins (fnil + 0.0) player-result)
                                           (update :visits (fnil + 0) 1))
                            t' (assoc-in t current-path updated-node)
                            parent-path (cons :root p)
                            parent-node (get-in t' parent-path (new-node))
                            updated-parent (-> parent-node
                                             (update :children assoc m updated-node))]
                        (assert (pos? (:visits updated-node)) "Path node must have positive visits")
                        (recur (assoc-in t' parent-path updated-parent)
                               current-path
                               more
                               (- r)
                               (- perspective)))))]
      ;; Always update the root node at the end
      (let [root (:root t-final)
            root-children (:children root)
            root-wins (:wins root 0.0)
            root-visits (:visits root 0)
            updated-root (-> (new-node)  ;; Create fresh node to avoid state issues
                          (assoc :wins (+ root-wins (if (pos? result) 1.0 0.0)))
                          (assoc :visits (inc root-visits))
                          (assoc :children (or root-children {})))]  ;; Ensure root has children
        (println "[backpropagate] Final root children count:" (count (:children updated-root)))
        (assoc t-final :root updated-root)))))

(defn simulate [game-state]
  "Run a random simulation from current state with improved move selection."
  (loop [state game-state
         depth 0
         max-depth 30
         current-player (f/current-player state)
         last-move nil
         moves-made []]
    (cond
      (f/game-over? state) 
      (let [result (if (= current-player :alice) 1.0 -1.0)]
        (debug-log "[DEBUG] Sim: depth=" depth ", result=" result ", moves=" (count moves-made))
        result)
      (>= depth max-depth) 
      (do
        (debug-log "[DEBUG] Sim: max depth=" depth ", moves=" (count moves-made))
        0.0)
      :else (let [moves (f/suggested-moves (:board state))
                  valid-moves moves]
              (if (empty? valid-moves)
                (let [result (if (= current-player :alice) -1.0 1.0)]
                  (debug-log "[DEBUG] Sim: no moves at depth=" depth ", result=" result)
                  result)
                (let [move (if (and last-move (> (count valid-moves) 1))
                            (rand-nth (remove #(= % last-move) valid-moves))
                            (rand-nth valid-moves))
                      new-state (f/make-move state move)]
                  (if new-state
                    (recur new-state 
                           (inc depth) 
                           max-depth 
                           (if (= current-player :alice) :bob :alice)
                           move
                           (conj moves-made move))
                    (let [result (if (= current-player :alice) -1.0 1.0)]
                      (debug-log "[DEBUG] Sim: invalid move at depth=" depth ", result=" result)
                      result))))))))

(defn best-move
  "Select the best move from a given trie and node path.
   Args:
   - trie: The MCTS search tree
   - path: The path to the node in the trie (default: root [])
   - game-state: The current game state (optional)"
  ([trie]
   (best-move trie [] (f/new-game)))
  ([trie path]
   (best-move trie path (f/new-game)))
  ([trie path game-state]
   (let [node (get-in trie path (new-node))
         children (:children node)]
     (println "[best-move] Path:" path)
     (println "[best-move] Node:" node)
     (println "[best-move] Children count:" (count children))
     (if (empty? children)
       (let [legal-moves (f/suggested-moves (:board game-state))
             compressed-moves (->> legal-moves
                                (map compress-move)
                                (take 50))
             children-map (into {} (map (fn [move] [move (new-node)]) compressed-moves))
             updated-node (assoc node :children children-map)
             updated-trie (assoc-in trie path updated-node)
             sorted-moves (->> children-map
                            (sort-by (fn [[_ {:keys [visits wins]}]]
                                     (/ wins (max 1 visits)))
                                   >))
             [best-move _] (first sorted-moves)]
         (when best-move
           (let [decompressed (decompress-move best-move)]
             (when (and (vector? decompressed)
                       (= 3 (count decompressed))
                       (f/valid-move? (:board game-state) decompressed))
               decompressed))))
       (let [sorted-moves (->> children
                            (sort-by (fn [[_ {:keys [visits wins]}]]
                                     (/ wins (max 1 visits)))
                                   >))
             [best-move _] (first sorted-moves)]
         (when best-move
           (let [decompressed (decompress-move best-move)]
             (when (and (vector? decompressed)
                       (= 3 (count decompressed))
                       (f/valid-move? (:board game-state) decompressed))
               decompressed))))))))

(defn get-node-stats [node]
  "Get statistics for a node"
  {:wins (:wins node 0.0)
   :visits (:visits node 0)
   :children-count (count (:children node))})

(defn inspect-trie [trie path]
  "Inspect a node in the trie"
  (let [node (get-in trie path)
        stats (get-node-stats node)]
    (println "Node at path" path ":")
    (println "Wins:" (:wins stats) "| Visits:" (:visits stats) "| Children:" (:children-count stats))
    (when-let [children (:children node)]
      (println "Top 3 children:")
      (doseq [[move child] (take 3 (sort-by (comp :visits second) > children))]
        (let [child-stats (get-node-stats child)]
          (println "Move:" move "-> Wins:" (:wins child-stats) "Visits:" (:visits child-stats)))))))

(defn mcts [initial-game-state iterations]
  "Main MCTS function"
  (let [initial-game-state (if (map? initial-game-state)
                            (f/->GameState (:board initial-game-state) 0)
                            initial-game-state)
        initial-tree (let [legal-moves (f/suggested-moves (:board initial-game-state))
                          compressed-moves (->> legal-moves
                                             (map compress-move)
                                             (take 50))
                          children-map (into {} (map (fn [move] [move (new-node)]) compressed-moves))
                          root-node (-> (new-node)
                                     (assoc :children children-map)
                                     (assoc :visits 1))]  ;; Initialize root with 1 visit
                      {:root root-node})]
    (debug-log "[DEBUG] MCTS: starting " iterations " iterations")
    (loop [tree initial-tree
           i 0]
      (when (zero? (mod i 100))
        (let [root (:root tree)]
          (debug-log (format "Iter %d | Root: %.1f/%d | Children: %d"
                           i
                           (:wins root)
                           (:visits root)
                           (count (:children root))))))
      (if (>= i iterations)
        (do
          (let [root (:root tree)]
            (assert (>= (:visits root) iterations) "Root visits must be >= iterations")
            (assert (pos? (count (:children root))) "Root must have children"))
          tree)
        (let [path (select-path tree initial-game-state)
              game-state (reduce (fn [gs m]
                                  (f/make-move gs (if (integer? m) (decompress-move m) m)))
                                initial-game-state
                                path)
              node (get-in tree (cons :root path))
              need-expand (and game-state
                              (not (f/game-over? game-state))
                              (empty? (:children node)))
              [tree path game-state] (if need-expand
                                      (let [expanded-tree (expand-node tree path game-state)]
                                        (if expanded-tree
                                          (let [moves (keys (:children (get-in expanded-tree (cons :root path))))
                                                first-move (first moves)
                                                new-state (when (and game-state 
                                                                   first-move
                                                                   (f/valid-move? (:board game-state) first-move))
                                                           (f/make-move game-state first-move))]
                                            [expanded-tree
                                             (conj path first-move)
                                             (or new-state game-state)])
                                          [tree path game-state]))
                                      [tree path game-state])
              result (when game-state (simulate game-state))
              updated-tree (if result
                            (let [backprop-tree (backpropagate tree path result)
                                  root (:root backprop-tree)
                                  root-children (:children root)
                                  root-wins (:wins root 0.0)
                                  root-visits (:visits root 0)
                                  updated-root (-> (new-node)  ;; Create fresh node to avoid state issues
                                                (assoc :wins root-wins)
                                                (assoc :visits (inc root-visits))
                                                (assoc :children (or root-children {})))]  ;; Ensure root has children
                              (assoc backprop-tree :root updated-root))
                            tree)]
          ;; Ensure all nodes in the path have children
          (let [final-tree (loop [t updated-tree
                                 p []
                                 [m & more] path
                                 state initial-game-state]
                            (if (nil? m)
                              t
                              (let [current-path (cons :root (conj p m))
                                    node (get-in t current-path)
                                    children (:children node)
                                    move-vec (when m (decompress-move m))
                                    new-state (when (and state move-vec)
                                               (f/make-move state move-vec))]
                                (if (empty? children)
                                  (let [legal-moves (when new-state
                                                    (f/suggested-moves (:board new-state)))
                                        compressed-moves (->> legal-moves
                                                           (map compress-move)
                                                           (take 50))
                                        children-map (into {} (map (fn [move] [move (new-node)]) compressed-moves))
                                        updated-node (assoc node :children children-map)
                                        t' (assoc-in t current-path updated-node)]
                                    (recur t' current-path more new-state))
                                  (recur t current-path more new-state)))))]
            ;; Ensure the root node has children and visits
            (let [root (:root final-tree)
                  root-children (:children root)
                  root-wins (:wins root 0.0)
                  root-visits (:visits root 0)]
              (if (or (empty? root-children) (zero? root-visits))
                (let [legal-moves (f/suggested-moves (:board initial-game-state))
                      compressed-moves (->> legal-moves
                                         (map compress-move)
                                         (take 50))
                      children-map (into {} (map (fn [move] [move (new-node)]) compressed-moves))
                      updated-root (-> (new-node)  ;; Create fresh node to avoid state issues
                                    (assoc :wins root-wins)
                                    (assoc :visits (inc root-visits))
                                    (assoc :children children-map))
                      final-tree' (assoc final-tree :root updated-root)]
                  (recur final-tree' (inc i)))
                (let [updated-root (-> (new-node)  ;; Create fresh node to avoid state issues
                                    (assoc :wins root-wins)
                                    (assoc :visits (inc root-visits))
                                    (assoc :children root-children))
                      final-tree' (assoc final-tree :root updated-root)]
                  (recur final-tree' (inc i)))))))))))


