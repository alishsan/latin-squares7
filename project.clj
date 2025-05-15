(defproject latin-squares7 "0.1.0"
  :description "7x7 Latin Squares Game"
  :dependencies [
    [org.clojure/clojure "1.11.1"]
;    [thinktopic/cortex "0.9.22"  :exclusions [org.bytedeco.javacpp-presets/cuda]]
    [thinktopic/cortex "0.9.22":exclusions [org.bytedeco.javacpp-presets/cuda] ]
    [net.mikera/core.matrix "0.62.0"]
    [org.clojure/math.combinatorics "0.1.6"]]
  :main latin-squares7.core
:pedantic? false  ; or :warn or abort
  :repl-options {:init-ns latin-squares7.core}
  :profiles {:dev {:dependencies [[org.clojure/tools.namespace "1.3.0"]]}})
