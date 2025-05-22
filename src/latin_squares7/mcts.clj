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

(defn select-path [trie game-state]
  "Select a path through the tree using UCB1"
  (loop [path []
         node (get trie [] (new-node))
         state game-state
         depth 0]
    (if (or (empty? (:children node))
            (f/game-over? state)
            (nil? state)
            (>= depth 10))  ;; Limit search depth
      path
      (let [total-visits (reduce + (map :visits (vals (:children node))))
            ;; Sort moves by UCB1 value with temperature-based selection
            moves (->> (:children node)
                    (map (fn [[m n]] 
                          [m (ucb1 n total-visits)]))
                    (sort-by second >))
            ;; Use temperature-based selection for more diversity
            temperature 0.5
            move-probs (->> moves
                          (map (fn [[m score]]
                                [m (math/exp (/ score temperature))]))
                          (sort-by second >))
            total-prob (reduce + (map second move-probs))
            normalized-probs (map (fn [[m p]] [m (/ p total-prob)]) move-probs)
            ;; Select move based on probabilities
            r (rand)
            [best-move _] (loop [[[m p] & more] normalized-probs
                               cum-prob 0.0]
                           (if (or (nil? m) (>= cum-prob r))
                             [m p]
                             (recur more (+ cum-prob p))))
            new-state (when (and state best-move (f/valid-move? (:board state) best-move))
                       (f/make-move state best-move))]
        (if new-state
          (recur (conj path best-move) 
                 (get (:children node) best-move) 
                 new-state
                 (inc depth))
          path)))))

(defn expand-node [trie path game-state]
  "Expand the tree by adding legal moves and update parent's :children map"
  (let [legal-moves (f/suggested-moves (:board game-state))]
    (when (empty? path)
      (debug-log "[DEBUG] Root: " (count legal-moves) " moves available"))
    (if (seq legal-moves)
      (let [moves (take 50 (shuffle legal-moves))
            new-nodes (into {} (map (fn [move] [move (new-node)]) moves))
            parent-node (if (empty? path)
                         (get trie [] (new-node))
                         (get-in trie path (new-node)))
            updated-parent (->Node
                           (:wins parent-node)
                           (:visits parent-node)
                           (:prior parent-node)
                           new-nodes)
            updated-trie (if (empty? path)
                          (assoc trie [] updated-parent)
                          (assoc-in trie path updated-parent))]
        ;; Verify the update
        (when (empty? path)
          (let [root (get updated-trie [])]
            (assert (pos? (count (:children root))) "Root node must have children after expansion")
            (assert (every? #(f/valid-move? (:board game-state) %) (keys (:children root))) 
                    "All children must be valid moves")))
        updated-trie)
      ;; If no legal moves, return trie with empty children
      (let [parent-node (if (empty? path)
                         (get trie [] (new-node))
                         (get-in trie path (new-node)))
            updated-parent (->Node
                           (:wins parent-node)
                           (:visits parent-node)
                           (:prior parent-node)
                           {})
            updated-trie (if (empty? path)
                          (assoc trie [] updated-parent)
                          (assoc-in trie path updated-parent))]
        updated-trie))))

(defn backpropagate [trie path result]
  "Backpropagate results through the tree and update node statistics."
  (let [result (double result)]
    (let [t-final (loop [t trie
                         p []
                         [m & more] path
                         r result
                         perspective 1.0]
                    (if (nil? m)
                      t
                      (let [node (get-in t (conj p m) (new-node))
                            player-result (if (= perspective 1.0)
                                          (if (pos? r) 1.0 0.0)
                                          (if (neg? r) 1.0 0.0))
                            updated-node (-> node
                                           (update :wins + player-result)
                                           (update :visits inc))
                            t' (assoc-in t (conj p m) updated-node)
                            parent-path p
                            parent-node (get-in t' parent-path (new-node))
                            updated-parent (-> parent-node
                                             (update :children assoc m updated-node))]
                        (assert (pos? (:visits updated-node)) "Path node must have positive visits")
                        (recur (assoc-in t' parent-path updated-parent)
                               (conj p m)
                               more
                               (- r)
                               (- perspective)))))]
      ;; Always update the root node at the end
      (let [root (get t-final [] (new-node))
            updated-root (-> root
                           (update :wins + (if (pos? result) 1.0 0.0))
                           (update :visits inc))]
        (assoc t-final [] updated-root)))))

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

(defn mcts [initial-game-state iterations]
  "Main MCTS function"
  (let [initial-game-state (if (map? initial-game-state)
                            (f/->GameState (:board initial-game-state) 0)
                            initial-game-state)]
    (debug-log "[DEBUG] MCTS: starting " iterations " iterations")
    (loop [trie {[] (new-node)}
           i 0]
      (when (zero? (mod i 100))
        (let [root (get trie [])]
          (debug-log (format "Iter %d | Nodes: %d | Root: %.1f/%d | Children: %d"
                           i
                           (count (keys trie))
                           (:wins root)
                           (:visits root)
                           (count (:children root))))))
      
      (if (>= i iterations)
        (do
          (let [root (get trie [])]
            (assert (>= (:visits root) iterations) "Root visits must be >= iterations")
            (assert (pos? (count (:children root))) "Root must have children"))
          trie)
        (let [trie (if (empty? (:children (get trie [])))
                    (let [expanded-trie (expand-node trie [] initial-game-state)]
                      (if expanded-trie
                        (do
                          (assert (pos? (count (:children (get expanded-trie [])))) 
                                  "Root must have children after expansion")
                          expanded-trie)
                        trie))
                    trie)
              path (select-path trie initial-game-state)
              node (get-in trie path)
              game-state (reduce f/make-move initial-game-state path)
              need-expand (and game-state
                              (not (f/game-over? game-state))
                              (empty? (:children node)))
              [trie path game-state] (if need-expand
                                      (let [expanded-trie (expand-node trie path game-state)]
                                        (if expanded-trie
                                          (let [moves (keys (:children (get-in expanded-trie path)))
                                                first-move (first moves)
                                                new-state (when (and game-state 
                                                                   first-move
                                                                   (f/valid-move? (:board game-state) first-move))
                                                           (f/make-move game-state first-move))]
                                            [expanded-trie
                                             (conj path first-move)
                                             (or new-state game-state)])
                                          [trie path game-state]))
                                      [trie path game-state])
              result (when game-state (simulate game-state))
              updated-trie (if result
                            (backpropagate trie path result)
                            trie)]
          (recur updated-trie (inc i)))))))

(defn best-move
  "Select the best move for the current game state using MCTS
   Args:
   - game-state: The current state of the game
   - iterations: Number of MCTS iterations to run (default: 1000)"
  ([game-state]
   (best-move game-state 1000))
  ([game-state iterations]
   (when (and game-state (not (f/game-over? game-state)))
     (let [trie (mcts game-state iterations)
           root-node (get trie [])
           children (:children root-node)]
       (when (seq children)
         (let [[best-move _] (->> children
                               (sort-by (fn [[_ {:keys [visits wins]}]] 
                                        (/ wins (max 1 visits)))
                                      >)
                               first)]
           (when (and best-move (f/valid-move? (:board game-state) best-move))
             (assert (vector? best-move) "Best move must be a vector")
             (assert (= 3 (count best-move)) "Best move must have length 3")
             best-move)))))))

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
