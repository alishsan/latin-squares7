(ns latin-squares7.core
  (:require [functions :as f]
            [latin-squares7.mcts :as mcts]
            [clojure.string :as str]))

(defn display-board [game-state]
  (println "\nCurrent board:")
  (f/print-board (:board game-state))
  (println "Current player:" (f/current-player game-state)))

(defn get-valid-move [game-state input]
  (try
    (let [move (read-string input)]
      (cond
        (not (vector? move)) (do (println "Move must be a vector") nil)
        (not= 3 (count move)) (do (println "Move needs 3 numbers") nil)
        :else (if (f/valid-move? (:board game-state) move)
                move
                (do (println "Invalid move! Check row/column constraints")
                    nil))))
    (catch Exception e
      (println "Error reading move:" (.getMessage e))
      nil)))

(defn handle-ai-move [game-state]
  (let [trie (mcts/mcts game-state 500)
        move (mcts/best-move trie)]
    (if move
      (do
        (println "\nAI plays:" move)
        (f/make-move game-state move))
      (do
        (println "AI cannot find valid move!")
        game-state))))

(defn auto-play-full-game []
  (loop [game-state (f/new-game)
         move-count 0]
    (println "\n=== Move" move-count "===")
    (f/print-board (:board game-state))
    
    (if (f/game-over? game-state)
      (do (println "Game over! Winner:" 
                   (if (= :alice (f/current-player game-state)) "Bob" "Alice"))
          game-state)
      (let [trie (mcts/mcts game-state 2000) ; 2000 iterations
            move (mcts/best-move trie)]
        (if move
          (recur (f/make-move game-state move) (inc move-count))
          (do (println "No valid moves left!")
              game-state))))))

(defn game-loop [game-state]
  (display-board game-state)
  (when (f/game-over? game-state)
    (println "Game over! Final board:")
    (f/print-board (:board game-state))
    (System/exit 0))
  
  (println "\nPossible moves:" (take 10 (f/suggested-moves (:board game-state))))
  (println "\nOptions:")
  (println "1. Enter move as [row col num]")
  (println "2. Type 'auto' for AI move")
  (println "3. Type 'quit' to exit")
  
  (let [input (read-line)]
    (cond
      (= input "quit") (do (println "Game over!") (System/exit 0))
      (= input "auto") (recur (handle-ai-move game-state))
      :else (if-let [move (get-valid-move game-state input)]
              (recur (f/make-move game-state move))
              (recur game-state)))))

(defn -main []
  (println "Latin Squares Game - 7x7 Board")
  (println "Players alternate placing numbers 1-7 without repeating in rows/columns")
  (game-loop (f/new-game)))

