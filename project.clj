(defproject latin-squares7 "0.1.0-SNAPSHOT"
  :description "Latin Squares game with AI"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [thinktopic/cortex "0.9.22" :exclusions [org.bytedeco.javacpp-presets/cuda]]
                 [metosin/malli "0.11.0"]
                 [techascent/tech.ml.dataset "7.039"]
                 [scicloj/scicloj.ml "0.3"]
                 [com.github.haifengl/smile-core "4.2.0"]
                 [techascent/tech.io "4.31"]
                 [cnuernber/dtype-next "10.128"]
                 [com.cnuernber/ham-fisted "2.020"]
                 [com.cnuernber/charred "1.034"]
                 [techascent/tech.resource "5.07"]
                 [org.clojure/math.combinatorics "0.2.0"]]
  :main ^:skip-aot latin-squares7.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all
                       :jvm-opts ["-Dclojure.compiler.direct-linking=true"]}})
