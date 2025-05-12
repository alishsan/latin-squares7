(ns latin-square7.core
  (:use functions))

(defn -main []
  (println "Starting Alice & Bob Game Simulation")
  (simulate-game)
  (println "\nMock Neural Network Evaluation Sample:")
  (println (mock-neural-evaluate (empty-board) :alice)))
