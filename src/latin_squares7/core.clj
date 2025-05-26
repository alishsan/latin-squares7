(ns latin-squares7.core
  (:require [latin-squares7.functions :as f]
            [latin-squares7.mcts :as mcts]
            [latin-squares7.nn-metamorph :as nn]
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

(defn handle-ai-move [game-state ai-type]
  (case ai-type
    :mcts (let [move (mcts/mcts game-state 500)]
            (if move
              (do
                (println "\nMCTS AI plays:" move)
                (f/make-move game-state move))
              (do
                (println "MCTS AI cannot find valid move!")
                game-state)))
    :neural (let [move (nn/get-best-move game-state)]
              (if move
                (do
                  (println "\nNeural AI plays:" move)
                  (f/make-move game-state move))
                (do
                  (println "Neural AI cannot find valid move!")
                  game-state)))))

(defn auto-play-from-position [game-state max-moves]
  "Autoplay from a given position using the neural network model"
  (loop [state game-state
         move-count 0]
    (println "\n=== Move" move-count "===")
    (f/print-board (:board state))
    
    (if (or (>= move-count max-moves)
            (f/game-over? state))
      (do (println "Game over! Final board:")
          (f/print-board (:board state))
          (println "Moves made:" move-count)
          (println "Solved?" (f/solved? state))
          state)
      (let [move (nn/get-best-move state)]
        (if move
          (recur (f/make-move state move) (inc move-count))
          (do (println "No valid moves left!")
              state))))))

(defn game-loop [game-state]
  (display-board game-state)
  (when (f/game-over? game-state)
    (println "Game over! Final board:")
    (f/print-board (:board game-state))
    (System/exit 0))
  
  (println "\nPossible moves:" (take 10 (f/suggested-moves (:board game-state))))
  (println "\nOptions:")
  (println "1. Enter move as [row col num]")
  (println "2. Type 'mcts' for MCTS AI move")
  (println "3. Type 'neural' for Neural Network AI move")
  (println "4. Type 'autoplay' to start autoplay from current position")
  (println "5. Type 'quit' to exit")
  
  (let [input (read-line)]
    (cond
      (= input "quit") (do (println "Game over!") (System/exit 0))
      (= input "mcts") (recur (handle-ai-move game-state :mcts))
      (= input "neural") (recur (handle-ai-move game-state :neural))
      (= input "autoplay") (do (println "Starting autoplay from current position...")
                              (auto-play-from-position game-state 100)
                              (System/exit 0))
      :else (if-let [move (get-valid-move game-state input)]
              (recur (f/make-move game-state move))
              (recur game-state)))))

(defn -main []
  (println "Latin Squares Game - 7x7 Board")
  (println "Players alternate placing numbers 1-7 without repeating in rows/columns")
  (game-loop (f/new-game)))

