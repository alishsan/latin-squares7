(ns latin-squares7.core
  (:require [functions :as f]
            [latin-squares7.mcts :as mcts]  ; Add this line
            [clojure.string :as str]))


(defn -main []
  (println "Latin Squares MCTS Engine")
  (let [game (f/new-game)]
    (loop [game-state game]
      (println "Current board:")
      (f/print-board (:board game))
      (println "Possible moves:" (f/suggested-moves (:board game)))
      (println "Enter 'auto' for AI move or [row col num] to move:")
    
      (let [input (read-line)]
        (when-not (= input "quit")
;          (if (= input "auto"))
      
      ;    (let [trie (mcts/mcts game 1000)
      ;          move (mcts/best-move trie game)]
      ;      (println "AI plays:" move)
      ;      (recur (f/make-move game move)))
          (let [move (read-string input)]
            (if (f/valid-move? (:board game) move)
              (recur (f/make-move game move))
              (do (println "Invalid move!") (recur game))))))
      )
))


