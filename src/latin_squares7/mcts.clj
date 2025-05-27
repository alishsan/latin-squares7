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
  (.write writer (format "Node{w:%d v:%d p:%.1f c:%d}"
                        (long (:wins node))
                        (long (:visits node))
                        (double (:prior node))
                        (count (:children node)))))

(defn compress-move [[r c n]]
  (+ (* 100 r) (* 10 c) n))

(defn decompress-move [move-int]
  (when move-int
    [(quot move-int 100)
     (quot (mod move-int 100) 10)
     (mod move-int 10)]))

(defn new-node [state prior]
  (->Node 0 0 prior {} state))

(defn ucb1 [node parent-visits]
  (let [wins (:wins node)
        visits (:visits node)
        prior (:prior node)
        c-puct 2.0]  ; Increased exploration constant
    (if (zero? visits)
      Double/POSITIVE_INFINITY
      (+ (/ wins visits)
         (* c-puct prior (Math/sqrt (/ parent-visits (inc visits))))))))

(defn select-path
  "Select a path from root to a leaf node using UCB1"
  [tree]
  (let [root (:root tree)]
    (loop [node root
           path []
           moves []]
      (if (or (nil? node)
              (f/game-over? (:state node)))
        [(conj path node) moves]
        (let [children (:children node)]
          (if (empty? children)
            [(conj path node) moves]
            (let [best-child (apply max-key #(ucb1 (val %) (:visits node)) children)
                  [move child] best-child]
              (recur child 
                     (conj path node)
                     (conj moves move)))))))))

(defn expand-node
  "Expand a leaf node by adding all possible moves as children"
  [node policy]
  (let [state (:state node)
        moves (f/suggested-moves (:board state))]
    (if (empty? moves)
      node
      (let [children (into {}
                          (map (fn [move]
                                 (let [move-key (compress-move move)
                                       prior (get policy move 0.1)]  ; Default prior if not in policy
                                   [move-key 
                                    (new-node (f/make-move state move) prior)]))
                               moves))]  ; Removed shuffle to maintain move order
        (assoc node :children children)))))

(defn simulate
  "Simulate a random playout from the given state"
  [state]
  (loop [current-state state
         depth 0]
    (if (or (f/game-over? current-state)
            (>= depth 49))  ; Maximum possible moves in a 7x7 board
      (if (f/solved? current-state)
        1.0  ;; Win
        0.0)  ;; Loss
      (let [moves (f/suggested-moves (:board current-state))]
        (if (empty? moves)
          0.0  ;; Loss
          (let [move (rand-nth moves)
                next-state (f/make-move current-state move)]
            (recur next-state (inc depth))))))))

(defn backpropagate
  "Update node statistics along the path"
  [tree path result]
  (when (and (seq path) (number? result))
    (loop [current-path path
           current-tree @tree]
      (when (seq current-path)
        (let [node (first current-path)
              updated-node (-> node
                             (update :visits inc)
                             (update :wins + result))
              updated-tree (if (= node (:root current-tree))
                           (assoc current-tree :root updated-node)
                           (let [root (:root current-tree)
                                 children (:children root)
                                 updated-children (into {}
                                                      (map (fn [[move child]]
                                                             [move (if (= child node)
                                                                    updated-node
                                                                    child)])
                                                           children))]
                             (assoc current-tree :root (assoc root :children updated-children))))]
          (swap! tree (constantly updated-tree))
          (recur (rest current-path) updated-tree))))))

(defn mcts
  "Monte Carlo Tree Search with neural network policy"
  [state iterations policy]
  (let [initial-root (new-node state 1.0)
        tree (atom {:root initial-root})]
    
    ;; Initial expansion of root node
    (let [root (:root @tree)
          expanded-root (expand-node root policy)]
      (swap! tree assoc :root expanded-root))
    
    (dotimes [_ iterations]
      (let [[path moves] (select-path @tree)
            leaf-node (last path)
            expanded-node (if (and (empty? (:children leaf-node))
                                 (not (f/game-over? (:state leaf-node))))
                          (expand-node leaf-node policy)
                          leaf-node)
            result (simulate (:state expanded-node))]
        
        ;; Update the tree with the expanded node and its children
        (when (not= leaf-node expanded-node)
          (let [root (:root @tree)
                updated-root (if (= leaf-node root)
                             expanded-node
                             (let [last-move (last moves)
                                   updated-children (assoc (:children root) last-move expanded-node)]
                               (assoc root :children updated-children)))]
            (swap! tree assoc :root updated-root)))
        
        ;; Backpropagate through the path
        (backpropagate tree path result)))
    
    ;; Select best move based on visit counts and win rate
    (let [root (:root @tree)
          children (:children root)]
      (when (seq children)
        (let [best-move (first (first (sort-by (fn [[_ child]]
                                                (let [visits (:visits child)
                                                      wins (:wins child)]
                                                  (if (zero? visits)
                                                    0.0
                                                    (/ wins visits))))
                                              > children)))]
          (when best-move
            (decompress-move best-move)))))))

(defn auto-play-full-game []
  "Play a full game using pure MCTS"
  (loop [game-state (f/new-game)
         moves []
         move-count 0]
    (if (or (f/game-over? game-state)
            (>= move-count 49))  ; Maximum possible moves in a 7x7 board
      {:board (:board game-state)
       :moves moves
       :solved? (f/solved? game-state)
       :moves-made move-count}
      (let [move (mcts game-state 1000 {})]  ;; Increased iterations for better move selection
        (if move
          (do
            (println "Move" (inc move-count) ":" move)
            (recur (f/make-move game-state move) 
                   (conj moves move)
                   (inc move-count)))
          (do
            (println "No valid move found at move" (inc move-count))
            {:board (:board game-state)
             :moves moves
             :solved? (f/solved? game-state)
             :moves-made move-count}))))))

(defn debug-mcts [state iterations]
  (println "Starting MCTS with state:" state)
  (println "Board valid?" (f/valid-game-state? state))
  (println "Suggested moves:" (f/suggested-moves (:board state)))
  (let [result (mcts state iterations {})]  ;; Empty policy for pure MCTS
    (println "MCTS result:" result)
    result))