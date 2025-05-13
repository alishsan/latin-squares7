(ns functions
(:require [clojure.spec.alpha :as s]))

(defn new-board [] 
  (vec (repeat 7 (vec (repeat 7 nil))))
)

(defn get-row [board row]
  (filter some? (board row)))

(defn get-col [board col]
  (filter some? (map #(nth % col) board)))

(defn valid-number? [num]
  (and (integer? num)
       (<= 1 num 7)))


(defn valid-move? [board [row col num]]
  (and (<= 0 row 6)           ; row bounds check
       (<= 0 col 6)           ; col bounds check
       (<= 1 num 7)           ; number bounds check
       (nil? (get-in board [row col]))
       (not (some #{num} (get-row board row)))
       (not (some #{num} (get-col board col)))))

(defn make-move [board [row col num]]
  {:pre [(valid-move? board [row col num])]}
  (assoc-in board [row col] num))

;; Game specifications
(s/def ::number (s/and int? #(<= 1 % 7)))
(s/def ::cell (s/nilable ::number))
(s/def ::row (s/coll-of ::cell :count 7))
(s/def ::board (s/coll-of ::row :count 7))
(s/def ::move (s/tuple int? int? ::number)) ; [row col num]

(defrecord GameState [board turn-number])

(defn new-game []
  (->GameState (new-board) 0))

(defn current-player [game]
  (if (even? (:turn-number game)) :alice :bob))

(defn make-move [game [row col num]]
  {:pre [(s/valid? ::move [row col num])
         (zero? (get-in (:board game) [row col]))]}
  (-> game
      (update :board #(assoc-in % [row col] num))
      (update :turn-number inc)))

(defn available-numbers [board]
  (let [used (set (filter some? (flatten board)))]
    (remove used (range 1 8))))

(defn suggested-moves [board]
  (for [row (range 7)
        col (range 7)
        :when (nil? (get-in board [row col]))
        num (available-numbers board)
        :when (valid-move? board [ row col num])]
    [row col num]))
