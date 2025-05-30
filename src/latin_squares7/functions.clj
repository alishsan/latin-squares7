(ns latin-squares7.functions
  (:require [clojure.spec.alpha :as s]
            [clojure.java.io :as io]))

;; ======================
;; Game Specifications
;; ======================
(s/def ::number (s/and int? #(<= 1 % 7)))
(s/def ::row-index (s/and int? #(<= 0 % 6)))
(s/def ::col-index (s/and int? #(<= 0 % 6)))
(s/def ::cell (s/nilable ::number))
(s/def ::row (s/coll-of ::cell :count 7))
(s/def ::board (s/coll-of ::row :count 7))
(s/def ::move (s/and vector?
                    #(= 3 (count %))
                    (s/cat :row ::row-index
                           :col ::col-index
                           :num ::number)))  ;; Strict number validation

;; ======================
;; Core Game Functions
;; ======================
(defn get-row [board row]
  (into #{} (filter some? (board row))))

(defn get-col [board col]
  (into #{} (filter some? (map #(nth % col) board))))

(defn new-board [] 
  (vec (repeat 7 (vec (repeat 7 nil)))))

(defn valid-number? [num]
  (and (integer? num)
       (<= 1 num 7)))

(defn valid-move? [board move]
  (if (and board (vector? move) (= 3 (count move)))
    (let [[row col num] move
          valid? (and (<= 0 row 6)
                      (<= 0 col 6)
                      (<= 1 num 7)
                      (nil? (get-in board [row col]))
                      (not-any? #(= num %) (get board row))
                      (not-any? #(= num %) (map #(get % col) board)))]
      (when-not valid?
        (println (format "Invalid move [%d %d %d] because:" row col num))
        (cond
          (not (<= 0 row 6)) (println "- Row out of bounds")
          (not (<= 0 col 6)) (println "- Column out of bounds")
          (not (<= 1 num 7)) (println "- Number out of range")
          (some? (get-in board [row col])) (println "- Cell is occupied")
          (some #(= num %) (get board row)) (println "- Number exists in row")
          (some #(= num %) (map #(get % col) board)) (println "- Number exists in column")))
      valid?)
    (do (println "Invalid move (nil or not a vector of length 3)") false)))

;; ======================
;; Game State Management
;; ======================
(defrecord GameState [board turn-number])

(defn new-game []
  {:board (vec (repeat 7 (vec (repeat 7 nil))))
   :current-player :alice})

(defn current-player [game-state]
  (:current-player game-state))

(defn make-move [game-state move]
  (when (and game-state move)  ; Return nil if either input is nil
    (let [board (:board game-state)
          [row col num] move]
      (when (valid-move? board move)  ; Only proceed if move is valid
        (let [new-board (assoc-in board [row col] num)
              new-player (if (= :alice (:current-player game-state))
                          :bob
                          :alice)
              turn-number (inc (or (:turn-number game-state) 0))]
          {:board new-board
           :current-player new-player
           :turn-number turn-number})))))

(defn valid-game-state? [game-state]
  (and (instance? GameState game-state)
       (s/valid? ::board (:board game-state))
       (integer? (:turn-number game-state))
       (>= (:turn-number game-state) 0)))

;; ======================
;; Move Analysis
;; ======================
(defn available-numbers [board]
  {:pre [(s/valid? ::board board)]}
  (let [used (set (filter some? (flatten board)))]
    (remove used (range 1 8))))

(defn valid-moves [board]
  (for [row (range 7)
        col (range 7)
        :when (nil? (get-in board [row col]))  ; Only consider empty cells
        num (range 1 8)
        :let [row-numbers (set (filter some? (get board row)))
              col-numbers (set (filter some? (map #(nth % col) board)))]
        :when (and (not (contains? row-numbers num))  ; Number not in row
                  (not (contains? col-numbers num)))]  ; Number not in column
    [row col num]))

(defn get-random-move [game-state]
  "Get a random valid move from the current position"
  (let [valid-moves (valid-moves (:board game-state))]
    (when (seq valid-moves)
      (rand-nth valid-moves))))

(defn game-over? [game-state]
  "Check if the game is over (either solved or blocked)"
  (let [board (:board game-state)
        is-full (every? #(every? some? %) board)
        valid-moves (valid-moves board)]
    (println "[DEBUG] Game over check:")
    (println "[DEBUG] Board is full?" is-full)
    (println "[DEBUG] Valid moves:" valid-moves)
    (println "[DEBUG] Number of valid moves:" (count valid-moves))
    (or is-full  ; Game is over if board is full
        (empty? valid-moves))))  ; Or if there are no valid moves

(defn solved? [game-state]
  "Check if the game is solved (board is full and all moves are valid)"
  (let [board (:board game-state)
        is-full (every? #(every? some? %) board)
        valid-moves (valid-moves board)]
    (and is-full  ; Board must be full
         (not (empty? valid-moves)))))  ; And must have valid moves

;; ======================
;; Move Compression
;; ======================
(defn compress-move [[r c n]]
  (+ (* 1000 r) (* 100 c) n))

(defn decompress-move [move-int]
  (when move-int
    [(quot move-int 1000)
     (quot (mod move-int 1000) 100)
     (mod move-int 100)]))

;; ======================
;; Game Play Functions
;; ======================
(defn play-game
  "Play a full game using the provided move selector function"
  [move-selector]
  (loop [game-state (new-game)
         moves []
         history []
         move-count 0]
    (if (or (game-over? game-state)
            (>= move-count 49))  ; Maximum possible moves in a 7x7 board
      {:board (:board game-state)
       :moves moves
       :history history
       :game-over? (game-over? game-state)
       :moves-made move-count}
      (let [move (move-selector game-state)]
        (if move
          (let [next-state (make-move game-state move)]
            (recur next-state
                   (conj moves move)
                   (conj history {:state game-state
                                :move move
                                :result (if (game-over? next-state) 1.0 0.0)})
                   (inc move-count)))
          {:board (:board game-state)
           :moves moves
           :history history
           :game-over? (game-over? game-state)
           :moves-made move-count})))))

(defn autoplay-from-position
  "Autoplay from a given position using the provided move selector function"
  [game-state max-moves move-selector]
  (loop [state game-state
         moves-made 0
         moves []]
    (if (or (>= moves-made max-moves)
            (game-over? state))
      {:final-state state
       :moves-made moves-made
       :game-over? (game-over? state)
       :moves moves}
      (let [move (move-selector state)]
        (if move
          (recur (make-move state move)
                 (inc moves-made)
                 (conj moves move))
          {:final-state state
           :moves-made moves-made
           :game-over? (game-over? state)
           :moves moves})))))

;; ======================
;; Debugging Utilities
;; ======================
(defn explain-move [board move]
  (when-not (valid-move? board move)
    (println "Invalid move because:")
    (cond
      (not (s/valid? ::move move)) (do
                                    (println "- Move format is invalid")
                                    (s/explain ::move move))
      (some? (get-in board [(first move) (second move)]))
        (println "- Cell is already occupied")
      (some #{(last move)} (get-row board (first move)))
        (println "- Number exists in row")
      (some #{(last move)} (get-col board (second move)))
        (println "- Number exists in column"))))

(defn debug-move-generation [board]
  (println "\n=== DEBUGGING MOVE GENERATION ===")
  (println "Board validation:" (s/valid? ::board board))
  (let [empty-cells (for [row (range 7)
                          col (range 7)
                          :when (nil? (get-in board [row col]))]
                      [row col])]
    (println "Empty cells:" (count empty-cells))
    (doseq [[row col] (take 5 empty-cells)]
      (let [row-numbers (set (filter some? (board row)))
            col-numbers (set (filter some? (map #(nth % col) board)))
            available (remove (into row-numbers col-numbers) (range 1 8))]
        (println (format "Cell [%d %d] - blocked by row: %s, col: %s | can take: %s"
                         row col
                         (or row-numbers "none")
                         (or col-numbers "none")
                         (seq available)))))))

;; ======================
;; Serialization
;; ======================
(defn game-state->edn [game-state]
  {:pre [(instance? GameState game-state)]}
  {:board (:board game-state)
   :turn-number (:turn-number game-state)})

(defn edn->game-state [{:keys [board turn-number]}]
  (->GameState board turn-number))

(defn save-game [game-state path]
  (io/make-parents path)
  (spit path (pr-str (game-state->edn game-state))))

(defn load-game [path]
  (-> path slurp read-string edn->game-state))

(defn print-board [board]
  (doseq [row board]
    (println (map #(or % "_") row))))

;; (println "Board valid?" (s/valid? ::board (:board (new-game))))
;; (println "Valid moves:" (valid-moves (:board (new-game))))
