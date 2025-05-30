(ns latin-squares7.core
  (:require [latin-squares7.functions :as f]
            [latin-squares7.nn-metamorph :as nn]
            [clojure.string :as str]))

(def ^:private game-state (atom (f/new-game)))
(def ^:private current-model (atom nil))
(def ^:private neural-rating (atom 1500))  ; Initial ELO rating
(def ^:private random-rating (atom 1500))  ; Initial ELO rating for random player

(defn initialize-model []
  "Initialize the neural network model if not already done"
  (when (nil? @current-model)
    (println "Initializing neural network model...")
    (reset! current-model (nn/create-game-pipeline))))

(defn get-best-move [state]
  "Get the best move using pure neural network predictions"
  (initialize-model)  ; Ensure model is initialized
  (let [predictions (nn/run-pipeline @current-model state :transform)
        policy (:policy predictions)
        valid-moves (f/valid-moves (:board state))
        move-number (count (filter some? (flatten (:board state))))]
    (println (format "[DEBUG] Move #%d" (inc move-number)))
    (println "[DEBUG] NN policy:" policy)
    (println "[DEBUG] Valid moves:" valid-moves)
    (let [move (when (seq valid-moves)
                 (apply max-key #(get policy % 0.0) valid-moves))]
      (println "[DEBUG] Chosen move:" move)
      move)))

(defn update-rating [rating opponent-rating result k-factor]
  "Update ELO rating based on game result
   result: 1.0 for win, 0.0 for loss, 0.5 for draw"
  (let [expected-score (/ 1.0 (+ 1.0 (Math/pow 10.0 (/ (- opponent-rating rating) 400.0))))
        rating-change (* k-factor (- result expected-score))]
    (+ rating rating-change)))

(defn get-random-move [game-state]
  "Get a random valid move from the current position"
  (let [valid-moves (f/valid-moves (:board game-state))]
    (when (seq valid-moves)
      (rand-nth valid-moves))))

(defn play-game [player1-move-fn player2-move-fn]
  "Play a game between two players, each using their move function"
  (loop [state (f/new-game)
         moves []]
    (if (f/game-over? state)
      {:final-state state
       :moves moves
       :winner (if (= (:current-player state) :alice) :bob :alice)}
      (let [move-fn (if (= (:current-player state) :alice)
                     player1-move-fn
                     player2-move-fn)
            move (move-fn state)]
        (if move
          (recur (f/make-move state move) (conj moves move))
          {:final-state state
           :moves moves
           :winner (if (= (:current-player state) :alice) :bob :alice)})))))

(defn evaluate-strength [n-games]
  "Evaluate neural network strength against random play"
  (println "\n=== Strength Evaluation ===")
  (println "Games to play:" n-games)
  (println "Initial neural network rating:" @neural-rating)
  (println "Initial random player rating:" @random-rating)
  (println "\nStarting games...")
  
  (let [k-factor 32  ; ELO K-factor for rating updates
        results (doall  ; Force evaluation of all games
                 (repeatedly n-games
                           (fn []
                             (let [game (play-game get-best-move get-random-move)
                                   winner (:winner game)
                                   neural-won? (= winner :alice)
                                   old-neural @neural-rating
                                   old-random @random-rating
                                   new-neural (update-rating old-neural old-random 
                                                           (if neural-won? 1.0 0.0) 
                                                           k-factor)
                                   new-random (update-rating old-random old-neural 
                                                           (if neural-won? 0.0 1.0) 
                                                           k-factor)]
                               (reset! neural-rating new-neural)
                               (reset! random-rating new-random)
                               {:winner winner
                                :neural-rating new-neural
                                :random-rating new-random
                                :moves (count (:moves game))}))))]
    (println "\n=== Final Results ===")
    (println "Games played:" n-games)
    (println "Final neural network rating:" @neural-rating)
    (println "Final random player rating:" @random-rating)
    
    (doseq [[i result] (map-indexed vector results)]
      (println (format "Game %d: %s won (neural: %d, random: %d, moves: %d)"
                      (inc i)
                      (name (:winner result))
                      (:neural-rating result)
                      (:random-rating result)
                      (:moves result))))
    
    {:neural-rating @neural-rating
     :random-rating @random-rating
     :results results}))

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

(defn handle-neural-move []
  "Handle move using pure neural network for any player"
  (when (not (f/game-over? @game-state))
    (let [move (get-best-move @game-state)]
      (when move
        (swap! game-state f/make-move move)
        (display-board @game-state)
        (when (f/game-over? @game-state)
          (println "Game over! Final board:")
          (f/print-board (:board @game-state)))))))

(defn auto-play-from-position [game-state max-moves]
  "Autoplay from a given position using the neural network model"
  (loop [state game-state
         moves-made 0
         moves []]
    (if (or (>= moves-made max-moves)
            (f/game-over? state))
      {:final-state state
       :moves-made moves-made
       :solved? (f/solved? state)
       :moves moves}
      (let [move (get-best-move state)]
        (if move
          (recur (f/make-move state move)
                 (inc moves-made)
                 (conj moves move))
          {:final-state state
           :moves-made moves-made
           :solved? (f/solved? state)
           :moves moves})))))

(defn game-loop []
  (display-board @game-state)
  (when (f/game-over? @game-state)
    (println "Game over! Final board:")
    (f/print-board (:board @game-state))
    (System/exit 0))
  
  (println "\nPossible moves:" (take 10 (f/valid-moves (:board @game-state))))
  (println "\nOptions:")
  (println "1. Enter move as [row col num]")
  (println "2. Type 'neural' for Neural Network AI move")
  (println "3. Type 'autoplay' to start autoplay from current position")
  (println "4. Type 'retrain' to retrain the model")
  (println "5. Type 'evaluate' to evaluate neural network strength")
  (println "6. Type 'quit' to exit")
  
  (let [input (read-line)]
    (cond
      (= input "quit") (do (println "Game over!") (System/exit 0))
      (= input "neural") (do (handle-neural-move) (recur))
      (= input "autoplay") (do (println "Starting autoplay from current position...")
                              (auto-play-from-position @game-state 100)
                              (System/exit 0))
      (= input "retrain") (do (println "Retraining model...")
                             (nn/retrain-model 50)  ; Retrain on 50 self-play games
                             (recur))
      (= input "evaluate") (do (println "Evaluating neural network strength...")
                              (evaluate-strength 20)  ; Evaluate with 20 games
                              (recur))
      :else (if-let [move (get-valid-move @game-state input)]
              (do (swap! game-state f/make-move move)
                  (recur))
              (recur)))))

(defn -main []
  (println "Latin Squares Game - 7x7 Board")
  (println "Players alternate placing numbers 1-7 without repeating in rows/columns")
  (game-loop))

