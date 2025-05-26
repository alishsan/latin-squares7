(ns latin-squares7.mcts
  (:require [latin-squares7.functions :as f]
            [clojure.math :as math]))

;; Debug logging setup
(def debug-log-file "mcts-debug.log")

(defn debug-log [msg]
  (println msg))

;; Node representation
(defrecord Node [wins visits prior children state])

;; Custom print-method for Node to make output more concise
(defmethod print-method Node [node writer]
  (.write writer (format "Node{w:%.1f v:%d p:%.1f c:%d}"
                        (:wins node)
                        (:visits node)
                        (:prior node)
                        (count (:children node)))))

(defn compress-move [[r c n]]
  (+ (* 1000 r) (* 100 c) n))

(defn decompress-move [move-int]
  [(quot move-int 1000)
   (quot (mod move-int 1000) 100)
   (mod move-int 100)])

(defn new-node [state prior]
  (->Node 0 0 prior {} state))

(defn ucb1 [node parent-visits]
  (let [wins (:wins node)
        visits (:visits node)
        prior (:prior node)]
    (if (zero? visits)
      Double/POSITIVE_INFINITY
      (+ (/ wins visits)
         (* prior (Math/sqrt (/ parent-visits (inc visits))))))))

(defn select-path
  "Select a path from root to a leaf node using UCB1"
  [tree]
  (loop [node (:root tree)
         path []]
    (if (or (nil? node)
            (empty? (:children node))
            (f/game-over? (:state node)))
      path
      (let [children (:children node)
            best-child (apply max-key #(ucb1 (val %) (:visits node)) children)
            [move child] best-child]
        (recur child (conj path [move child]))))))

(defn expand-node
  "Expand a leaf node by adding all possible moves as children"
  [node]
  (let [state (:state node)
        moves (f/suggested-moves (:board state))]
    (if (empty? moves)
      node
      (let [children (into {}
                          (map (fn [move]
                                 [(compress-move move) 
                                  (new-node (f/make-move state move) 1.0)])
                               moves))]
        (assoc node :children children)))))

(defn simulate
  "Simulate a random playout from the given state"
  [state]
  (loop [current-state state]
    (if (f/game-over? current-state)
      (if (f/solved? current-state)
        1.0  ;; Win
        0.0)  ;; Loss
      (let [moves (f/suggested-moves (:board current-state))]
        (if (empty? moves)
          0.0  ;; Loss
          (recur (f/make-move current-state (rand-nth moves))))))))

(defn backpropagate
  "Update node statistics along the path"
  [path result]
  (doseq [[move node] (reverse path)]
    (-> node
        (update :visits inc)
        (update :wins + result))))

(defn best-move
  "Select the best move based on visit counts"
  [tree]
  (when tree
    (let [root (:root tree)
          children (:children root)]
      (when (seq children)
        (let [best-child (apply max-key :visits (vals children))]
          (decompress-move (first (first (filter #(= (val %) best-child) children)))))))))

;; Debug functions
(defn get-node-stats [tree]
  (let [children (:children tree)]
    (map (fn [[move child]]
           [move 
            {:visits (:visits child)
             :wins (:wins child)
             :prior (:prior child)}])
         children)))

(defn inspect-tree [tree]
  (let [root (:root tree)]
    {:root {:wins (:wins root)
            :visits (:visits root)
            :prior (:prior root)
            :children (count (:children root))}
     :total-nodes (count (tree-seq :children :children root))}))

(defn mcts
  "Monte Carlo Tree Search"
  [state iterations]
  (let [initial-root (new-node state 1.0)  ;; Root node has prior 1.0
        tree (atom {:root initial-root})]
    
    ;; First expansion of root node
    (let [initial-moves (f/suggested-moves (:board state))
          initial-children (into {} (map (fn [move] 
                                         [(compress-move move)
                                          (new-node (f/make-move state move) 1.0)]) 
                                       initial-moves))
          root-with-children (assoc initial-root :children initial-children)]
      (reset! tree {:root root-with-children}))
    
    (dotimes [_ iterations]
      (let [path (select-path @tree)
            node (get-in @tree (cons :root path))
            expanded-path (if (and (empty? (:children node))
                                 (not (f/game-over? (:state node))))
                          (expand-node node)
                          path)
            expanded-node (get-in @tree (cons :root expanded-path))
            result (simulate (:state expanded-node))
            _ (backpropagate expanded-path result)]))
    
    (best-move @tree)))

(defn auto-play-full-game []
  "Play a full game using MCTS"
  (loop [game-state (f/new-game)
         moves []]
    (if (f/game-over? game-state)
      {:board (:board game-state)
       :moves moves
       :solved? (f/solved? game-state)}
      (let [move (mcts game-state 500)]  ;; Using 500 iterations for each move
        (if move
          (recur (f/make-move game-state move) (conj moves move))
          {:board (:board game-state)
           :moves moves
           :solved? (f/solved? game-state)})))))

;; Add debug logging to help diagnose issues
(defn debug-mcts [state iterations]
  (println "Starting MCTS with state:" state)
  (println "Board valid?" (f/valid-game-state? state))
  (println "Suggested moves:" (f/suggested-moves (:board state)))
  (let [result (mcts state iterations)]
    (println "MCTS result:" result)
    result))