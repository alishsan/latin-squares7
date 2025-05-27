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

(defn valid-move? [board [row col num]]
  (and (<= 0 row 6)
       (<= 0 col 6)
       (<= 1 num 7)
       (nil? (get-in board [row col]))
       (not-any? #(= num %) (get board row))
       (not-any? #(= num %) (map #(get % col) board))))

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
  (let [board (:board game-state)
        [row col num] move
        new-board (assoc-in board [row col] num)
        new-player (if (= :alice (:current-player game-state))
                    :bob
                    :alice)]
    {:board new-board
     :current-player new-player}))

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

(defn suggested-moves [board]
  (for [row (range 7)
        col (range 7)
        :when (nil? (get-in board [row col]))
        num (range 1 8)
        :when (valid-move? board [row col num])]
    [row col num]))

(defn solved? [game-state]
  (let [board (:board game-state)]
    (every? some? (flatten board))))

(defn game-over? [game-state]
  (let [board (:board game-state)]
    (or (every? some? (flatten board))
        (empty? (suggested-moves board)))))

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
;; (println "Suggested moves:" (suggested-moves (:board (new-game))))
