(ns latin-squares7.mcts
  (:require [latin-squares7.functions :as f]
            [clojure.math :as math]
            [clojure.java.io :as io]))

;; Debug logging setup
(def debug-log-file "mcts-debug.log")

(defn debug-log [msg]
  (println msg))

;; Node representation
(defrecord Node [wins visits prior children])

;; Custom print-method for Node to make output more concise
(defmethod print-method Node [node writer]
  (.write writer (format "Node{w:%.1f v:%d p:%.1f c:%d}"
                        (:wins node)
                        (:visits node)
                        (:prior node)
                        (count (:children node)))))

(defn new-node []
  {:wins 0
   :visits 0
   :children {}
   :state nil})

;; UCB1 formula for move selection
(defn ucb1 [node parent-visits]
  (let [visits (:visits node)
        wins (:wins node)
        c (Math/sqrt 2)]
    (if (zero? visits)
      Double/POSITIVE_INFINITY
      (+ (/ wins visits)
         (* c (Math/sqrt (/ (Math/log parent-visits) visits)))))))

;; Utility functions for move compression
(defn compress-move [[r c n]]
  (+ (* 100 r) (* 10 c) n))

(defn decompress-move [move-int]
  [(quot move-int 100)
   (quot (mod move-int 100) 10)
   (mod move-int 10)])

;; Update select-path to use compressed moves
(defn select-path
  "Select a path from root to a leaf node using UCB1"
  [tree]
  (loop [node tree
         path [tree]]
    (if (or (nil? node)
            (empty? (:children node))
            (f/game-over? (:state node)))
      path
      (let [children (:children node)
            best-child (apply max-key #(ucb1 (val %) (:visits node)) children)
            [move child] best-child]
        (recur child (conj path [move child]))))))

;; Update expand-node to use compressed moves as keys
(defn expand-node
  "Expand a leaf node by adding all possible moves as children"
  [node]
  (let [state (:state node)
        moves (f/suggested-moves (:board state))]
    (if (empty? moves)
      node
      (let [children (into {}
                          (map (fn [move]
                                 [move {:state (f/make-move state move)
                                       :visits 0
                                       :wins 0
                                       :children {}}])
                               moves))]
        (assoc node :children children)))))

;; Update backpropagate to use compressed moves in path
(defn backpropagate
  "Update node statistics along the path"
  [path result]
  (doseq [node (reverse path)]
    (-> node
        (update :visits inc)
        (update :wins + (if (= result :win) 1 0)))))

(defn simulate
  "Simulate a random playout from the given state"
  [state]
  (loop [current-state state]
    (if (f/game-over? current-state)
      (if (f/solved? current-state)
        :win
        :loss)
      (let [moves (f/suggested-moves (:board current-state))]
        (if (empty? moves)
          :loss
          (recur (f/make-move current-state (rand-nth moves))))))))

(defn best-move
  "Select the best move based on visit counts"
  ([tree]
   (when tree
     (let [root (:root tree)
           children (:children root)]
       (when (seq children)
         (let [best-child (apply max-key :visits (vals children))]
           (first (first (filter #(= (val %) best-child) children))))))))
  ([tree path]
   (when (and tree path)
     (let [node (get-in tree (cons :root path))
           children (:children node)]
       (when (seq children)
         (let [best-child (apply max-key :visits (vals children))]
           (first (first (filter #(= (val %) best-child) children)))))))))

(defn get-node-stats
  "Get statistics for a node in the tree"
  [tree]
  (let [children (:children tree)]
    (map (fn [[move child]]
           [move (:visits child) (:wins child)])
         children)))

(defn inspect-trie
  "Debug function to inspect the tree structure"
  [tree]
  (let [root (:root tree)]
    {:root {:wins (:wins root)
            :visits (:visits root)
            :children (count (:children root))}
     :total-nodes (count (tree-seq :children :children root))}))

(defn mcts
  "Monte Carlo Tree Search implementation"
  [state iterations]
  (when (f/valid-game-state? state)
    (let [initial-root {:wins 0
                       :visits 1  ; Start with 1 visit
                       :children {}
                       :state state}
          tree (atom {:root initial-root})]
     
      ;; First expansion of root node
      (let [initial-moves (map compress-move (f/suggested-moves (:board state)))
            initial-children (into {} (map (fn [move] 
                                           [move (assoc (new-node) 
                                                      :state (f/make-move state (decompress-move move)))]) 
                                         initial-moves))
            root-with-children (assoc initial-root :children initial-children)]
        (reset! tree {:root root-with-children}))
     
      (dotimes [_ iterations]
        (let [path (select-path @tree)
              node (get-in @tree (cons :root path))
              expanded-path (if (and (empty? (:children node))
                                   (not (f/solved? (:state node))))
                            (expand-node node)
                            path)
              expanded-node (get-in @tree (cons :root expanded-path))
              result (if (f/valid-game-state? (:state expanded-node))
                      (simulate (:state expanded-node))
                      0)  ; Return 0 if state is invalid
              _ (backpropagate expanded-path result)]))
     
      (let [final-tree @tree
            root (:root final-tree)]
        (when (seq (:children root))
          (let [best-move-int (best-move final-tree)]
            (when best-move-int
              (decompress-move best-move-int))))))))

;; Add debug logging to help diagnose issues
(defn debug-mcts [state iterations]
  (println "Starting MCTS with state:" state)
  (println "Board valid?" (f/valid-game-state? state))
  (println "Suggested moves:" (f/suggested-moves (:board state)))
  (let [result (mcts state iterations)]
    (println "MCTS result:" result)
    result))

(defn auto-play-full-game []
  "Play a full game using MCTS AI"
  (loop [game-state (f/new-game)
         move-count 0]
    (if (f/game-over? game-state)
      game-state
      (let [move (mcts game-state 2000)] ; 2000 iterations
        (if move
          (recur (f/make-move game-state move) (inc move-count))
          game-state)))))