(ns functions)

;; 1. Core Game Logic (unchanged)
(defn empty-board [] 
  (vec (repeat 7 (vec (repeat 7 nil)))))

(defn row-numbers [board row]
  (filter some? (board row)))

(defn col-numbers [board col]
  (filter some? (map #(nth % col) board)))

(defn valid-move? [board row col num]
  (and (nil? (get-in board [row col]))
       (not (some #{num} (row-numbers board row)))
       (not (some #{num} (col-numbers board col)))))

;; 2. Fixed Mock Move Selection
(defn get-legal-moves [board num]
  (for [row (range 7)
        col (range 7)
        :when (valid-move? board row col num)]
    [row col]))

(defn mock-select-move [state iterations]
  (let [num (inc (mod (:last-number state) 7))
        legal-moves (get-legal-moves (:board state) num)]
    (when (seq legal-moves)
      (rand-nth legal-moves))))


;; 3. Visualization (unchanged)
(defn print-board [board]
  (println "\n  0 1 2 3 4 5 6")
  (doseq [row (range 7)]
    (print row " ")
    (doseq [col (range 7)]
      (print (or (get-in board [row col]) ".") " "))
    (println)))


;; 4. Fixed Game Simulation
(defn simulate-game []
  (loop [state {:board (empty-board) 
                :player :alice
                :last-number 0}
         move-count 1]
    (print-board (:board state))
    (println (str "\n" (name (:player state)) "'s turn (Move " move-count ")"))
    
    (let [num (inc (mod (:last-number state) 7))
          move (mock-select-move state 100)]
      (if move
        (let [[row col] move
              new-board (assoc-in (:board state) [row col] num)]
          (println (str "Placing " num " at (" row "," col ")"))
          (recur {:board new-board
                  :player (if (= (:player state) :alice) :bob :alice)
                  :last-number num}
                 (inc move-count)))
        (println (str (name (:player state)) " cannot move. Game over!"))))))
