(ns latin-squares7.core
  (:require [latin-squares7.functions :as f]
            [latin-squares7.mcts :as mcts]
            [latin-squares7.nn-mcts :as nn-mcts]
            [latin-squares7.nn-metamorph :as nn]
            [clojure.string :as str]))

(def ^:private game-state (atom (f/new-game)))

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

(defn handle-ai-move []
  "Handle AI move using neural-guided MCTS"
  (when (and (not (f/game-over? @game-state))
             (= (:current-player @game-state) :ai))
    (let [move (nn-mcts/get-best-move @game-state)]
      (when move
        (swap! game-state f/make-move move)
        (display-board @game-state)
        (when (f/game-over? @game-state)
          (println "Game over! Final board:")
          (f/print-board (:board @game-state)))))))

(defn auto-play-from-position [game-state max-moves]
  "Autoplay from a given position using the neural network model"
  (nn-mcts/autoplay-from-position game-state max-moves))

(defn game-loop []
  (display-board @game-state)
  (when (f/game-over? @game-state)
    (println "Game over! Final board:")
    (f/print-board (:board @game-state))
    (System/exit 0))
  
  (println "\nPossible moves:" (take 10 (f/suggested-moves (:board @game-state))))
  (println "\nOptions:")
  (println "1. Enter move as [row col num]")
  (println "2. Type 'mcts' for Neural-guided MCTS AI move")
  (println "3. Type 'neural' for Pure Neural Network AI move")
  (println "4. Type 'autoplay' to start autoplay from current position")
  (println "5. Type 'retrain' to retrain the model")
  (println "6. Type 'quit' to exit")
  
  (let [input (read-line)]
    (cond
      (= input "quit") (do (println "Game over!") (System/exit 0))
      (= input "mcts") (do (handle-ai-move) (recur))
      (= input "neural") (do (handle-ai-move) (recur))
      (= input "autoplay") (do (println "Starting autoplay from current position...")
                              (auto-play-from-position @game-state 100)
                              (System/exit 0))
      (= input "retrain") (do (println "Retraining model...")
                             (nn/retrain-model 50)  ; Retrain on 50 self-play games
                             (recur))
      :else (if-let [move (get-valid-move @game-state input)]
              (do (swap! game-state f/make-move move)
                  (recur))
              (recur)))))

(defn -main []
  (println "Latin Squares Game - 7x7 Board")
  (println "Players alternate placing numbers 1-7 without repeating in rows/columns")
  (game-loop))

