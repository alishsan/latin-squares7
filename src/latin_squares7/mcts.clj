(ns latin-squares.mcts
  (:require [latin-squares.functions :as f]
            [clojure.set :as set]
            [clojure.string :as str]))

;; Trie structure will track:
;; Key: Sequence of moves [[row col num] [row col num] ...]
;; Value: {:wins X :visits Y}

(defn add-to-trie [trie path stats]
  (update-in trie path (fn [existing]
                         (merge-with + 
                                    {:wins 0 :visits 0} 
                                    existing 
                                    stats)))

(defn node-stats [trie path]
  (get-in trie path {:wins 0 :visits 0}))

(defn ucb1 [trie path total-visits]
  (let [{:keys [wins visits]} (node-stats trie path)
        exploitation (/ wins (max 1 visits))
        exploration (Math/sqrt (/ (Math/log total-visits) (max 1 visits)))]
    (+ exploitation (* 1.41 exploration)))) ; √2 ≈ 1.41

(defn select-node [trie current-path game-state]
  (let [valid-moves (f/suggested-moves (:board game-state))
        total-visits (:visits (node-stats trie current-path) 1)]
    (if (empty? valid-moves)
      current-path
      (let [best-move (->> valid-moves
                           (map #(conj current-path %))
                           (apply max-key #(ucb1 trie % total-visits)))]
        (if (zero? (get-in trie [best-move :visits] 0))
          best-move
          (select-node trie best-move (f/make-move game-state (last best-move)))))))

(defn simulate [game-state]
  ;; Random simulation returns 1 if current player wins
  (if (f/game-over? game-state)
    (if (= (f/current-player game-state) :alice) 1 0)
    (let [moves (f/suggested-moves (:board game-state))
          random-move (rand-nth moves)]
      (simulate (f/make-move game-state random-move)))))

(defn backpropagate [trie path result]
  (loop [t trie
         p path
         r result]
    (if (empty? p)
      t
      (recur (add-to-trie t p {:wins r :visits 1})
             (pop p)
             (- 1 r))))) ; Alternate win/loss for players

(defn mcts [initial-game-state iterations]
  (let [root-path []]
    (loop [trie {}
           i 0]
      (if (>= i iterations)
        trie
        (let [path (select-node trie root-path initial-game-state)
              game-state (reduce f/make-move initial-game-state path)
              result (simulate game-state)
              new-trie (backpropagate trie path result)]
          (recur new-trie (inc i)))))))

(defn best-move [trie game-state]
  (let [valid-moves (f/suggested-moves (:board game-state))
        move-stats (map #(assoc (node-stats trie [%]) :move %) valid-moves)
        total-visits (reduce + (map :visits move-stats))]
    (->> move-stats
         (sort-by #(/ (:wins %) (max 1 (:visits %))))
         last
         :move)))


  (defn parallel-mcts [game-state iterations threads]
  (let [iter-per-thread (quot iterations threads)
        futures (doall (repeatedly threads 
                       #(future (mcts game-state iter-per-thread))))]
    (reduce (partial merge-with (partial merge-with +)) 
            {} 
            (map deref futures))))

  (def memoized-simulate (memoize simulate))
  
;; Usage example:
;; (def initial-game (f/new-game))
;; (def trie (mcts initial-game 1000))
;; (best-move trie initial-game)
