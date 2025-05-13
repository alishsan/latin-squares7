(ns latin-squares7.core
  (:require [functions :as f])
  (:gen-class))

(defn -main []
  (let [board (f/new-board)
        move1 [0 0 1]
        move2 [0 1 2]]
    (println "Empty board:" board)
    (println "Valid move?" (f/valid-move? board 0 0 1))
    (let [updated (-> board
                    (f/make-move move1)
                    (f/make-move move2))]
      (println "After moves:" updated)
      (println "Game over?" (f/game-over? updated)))))
