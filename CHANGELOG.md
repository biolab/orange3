Change Log
==========

[next] - TBA
------------
* ...

[3.13.0] - 2018-04-17
--------------------
##### Enhancements
* canvas/add-ons: Add extra packages via a name input dialog ([#3006](../../pull/3006))
* Variable lists (with QListView) optimizations ([#2994](../../pull/2994))

##### Bugfixes
* Add-ons working again (PyPI JSON interface + local official index) ([#3005](../../pull/3005))
* Fix variable type guessing ([#2998](../../pull/2998))
* Addon dialog crashes when site-packages directory does not exist ([#2988](../../pull/2988))
* Fix reading double quoted text fields ([#2989](../../pull/2989))

[3.12.0] - 2018-04-06
--------------------
##### Enhancements
* owselectrows: Change defaults for 'Purging' ([#2969](../../pull/2969))
* statistics: Speed up countnans for sparse matrices ([#2965](../../pull/2965))

##### Bugfixes
* Sieve Diagram: Fix spacing of axis labels ([#2971](../../pull/2971))
* Fix data reading speed ([#2923](../../pull/2923))
* KMeans clear results on k change, do not recluster ([#2915](../../pull/2915))
* gui.ControlledList: Take a weakref to listBox ([#2962](../../pull/2962))
* WidgetManager: Schedule delayed deletion for managed OWWidgets ([#2963](../../pull/2963))
* domaineditor: Give the VarTableModel a parent ([#2961](../../pull/2961))
* scatterplot: limit number of points displayed in tooltip ([#2980](../../pull/2980))
* Speed-up prediction by circumventing Instance-specific prediction. ([#2959](../../pull/2959))
* Vizrank: Properly shutdown/wait when parent deleted ([#2960](../../pull/2960))
* Test & Score: Make the scores view table non-editable ([#2947](../../pull/2947))


[3.11.0] - 2018-03-07
--------------------
##### Enhancements
* Distances: Optimize PearsonR/SpearmanR ([#2852](../../pull/2852))
* Data Table: Optimize performance ([#2905](../../pull/2905))

##### Bugfixes
* Save Image to SVG fixed on Qt5 ([#2930](../../pull/2930))
* Test & Score: Resort scores when data changes ([#2931](../../pull/2931))
* distribution.py: Fix computation on multiclass data ([#2903](../../pull/2903))
* contingency.pyx: Fix out of bound write ([#2924](../../pull/2924))
* Test and Score: Fix averaging over classes for binary scores ([#2918](../../pull/2918))
* sgd: Change deprecated n_iter to max_iter ([#2920](../../pull/2920))
* heatmap: Do not crash on all zero column ([#2916](../../pull/2916))


[3.10.0] - 2018-02-19
--------------------
##### Enhancements
* Select Rows: Add annotated data output ([#2908](../../pull/2908))
* canvas: Open dropped ows files ([#2885](../../pull/2885))
* Settings for HTTP and HTTPS proxies in the canvas ([#2906](../../pull/2906))
* Add-ons: Option to list only trusted add-ons ([#2899](../../pull/2899))

##### Bugfixes
* SPG Legend: Fix vertical spacing ([#2914](../../pull/2914))


[3.9.1] - 2018-02-02
--------------------
##### Enhancements
* Add parameters and similarity measures to tSNE ([#2510](../../pull/2510))
* Canvas: Add zoom ([#2841](../../pull/2841))

##### Bugfixes
* OWWidget: Store quicktip displayed state in non versioned settings dir ([#2875](../../pull/2875))
* Impute: Fix state serialization/restore ([#2830](../../pull/2830))
* Feature Constructor: Make FeatureFunc picklable ([#2873](../../pull/2873))
* Projection widgets: transform data properly ([#2871](../../pull/2871))


[3.9.0] - 2018-01-19
--------------------
##### Enhancements
* Linear Discriminant Analysis: scripting part ([#2823](../../pull/2823))
* owdistances: Add 'Normalize' check box ([#2851](../../pull/2851))
* Variable: Simplify the is_{discrete,continuous,...} implementation ([#2874](../../pull/2874))
* manifold: Use arpack for decomposition in `torgerson` ([#2825](../../pull/2825))
* Radviz: new widget ([#2480](../../pull/2480))
* Canvas: Improve preview rendering ([#2784](../../pull/2784))
* Linear Projection (LDA, PCA) ([#2445](../../pull/2445))
* Scatter Plot Graph: max discrete values colors and shape ([#2804](../../pull/2804))
* Scatter Plot Graph: legend opacity ([#2819](../../pull/2819))

##### Bugfixes
* Add labels to combo when data comes from distance matrix ([#2866](../../pull/2866))
* utils/concurrent: Handle an empty futures list ([#2834](../../pull/2834))
* OWWidget: Move 'splitter' to private members ([#2847](../../pull/2847))
* Radviz: enable VizRank numeric color (class) ([#2853](../../pull/2853))
* Bincount: Fix crash on array with all nans ([#2831](../../pull/2831))
* mds: Fix incorrect assert ([#2844](../../pull/2844))
* VizRank (Linear Projection, Radviz): spin disabled/enabled ([#2846](../../pull/2846))
* Windows installers: Python lookup ([#2827](../../pull/2827))
* Canvas: Palette propagation ([#2760](../../pull/2760))
* mssql: Catch errors due to incorrect connection params ([#2838](../../pull/2838))
* canvas/addons: Fix progress dialog showing up when not necessary ([#2833](../../pull/2833))
* ContextHandler: Merge local context into globals before serialization ([#2837](../../pull/2837))
* Hierarchical Clustering: Fix size constraints ([#2796](../../pull/2796))
* canvas/annotationitem: Use separate item for shadow base ([#2821](../../pull/2821))
* Scatter Plot Graph: vars instead of indices & remove dead code ([#2815](../../pull/2815))
* Table: classes (Y) must be float ([#2822](../../pull/2822))


[3.8.0] - 2017-12-01
--------------------
##### Enhancements
* New signals: Trees, Forest ([#2801](../../pull/2801))
* Scatter Plot: Improve tooltips ([#2703](../../pull/2703))
* Allow custom (generic) names in Transpose Widget  ([#2737](../../pull/2737))
* Scatter Plot VizRank: some fixes and regard to color ([#2787](../../pull/2787))
* Improved Sparsity Handling ([#2341](../../pull/2341))
* Error Reporting: send report even when recursion error is raised
* test_owdatasets: Test files at different (dir tree) depths
* [FIX] Rank: should not fail on data with no attributes
* Domain: Add copy method ([#2734](../../pull/2734))
* Domain Model: order without separators ([#2697](../../pull/2697))

##### Bugfixes
* Test & Learn: do not crash on a data with class only nans ([#2751](../../pull/2751))
* FreeViz: 2 issues when no data ([#2780](../../pull/2780))
* [ENH] Scatter Plot VizRank: some fixes and regard to color ([#2787](../../pull/2787))
* Scatter Plot Graph: crash on metas column with all 0 values ([#2775](../../pull/2775))
* Scatter Plot: subset data ([#2773](../../pull/2773))
* Scatter Plot: VizRank disabled when no class vars ([#2757](../../pull/2757))
* Error Reporting: send report even when recursion error is raised
* Select Rows: None on output when no data ([#2726](../../pull/2726))
* test_owdatasets: Test files at different (dir tree) depths
* [FIX] Rank: should not fail on data with no attributes
* Predictions: space added before bracket in meta name. ([#2742](../../pull/2742))
* Fix AbstractSortTableModel.mapFromSourceRows for empty list or array ([#2730](../../pull/2730))
* Correspondence Analysis: do not crash when no categorical ([#2723](../../pull/2723))
* ScatterPlotGraph: fix zoom CTRL + and CTRL - ([#2716](../../pull/2716))
* errorreporting: Remove use of pip internal api ([#2724](../../pull/2724))


[3.7.1] - 2017-11-17
--------------------
##### Enhancements
* MDS: Support showing textual features as labels ([#2699](../../pull/2699))

##### Bugfixes
* canvas/canvasmain: Fix 'Examples' action handling in the welcome dialog ([#2779](../../pull/2779))
* Nomogram on PyQt4 ([#2763](../../pull/2763))
* Broken installation of Installation of wheels ([#2765](../../pull/2765))
* Add-on installation crashes (when conda not in PATH) ([#2725](../../pull/2725))


[3.7.0] - 2017-10-27
--------------------
##### Enhancements
* Data Sets: Add filter ([#2695](../../pull/2695))
* Add-on installation with Conda ([#2561](../../pull/2561))
* Add Groups column to Selected Data in Scatter plot output ([#2678](../../pull/2678))
* DomainModel: Don't Show Hidden Variables by Default ([#2690](../../pull/2690))
* FreeViz: new widget ([#2512](../../pull/2512))
* FreeViz script ([#2563](../../pull/2563))
* Boxplot: Allow hiding labels ([#2654](../../pull/2654))
* owmds: Support selection/output of multiple groups ([#2666](../../pull/2666))
* Widget status bar buttons ([#2514](../../pull/2514))
* owfile: allow multiple readers with same extension ([#2644](../../pull/2644))

##### Bugfixes
* Tree Viewer: reset view to top left ([#2705](../../pull/2705))
* ScatterPlot Crashes on Data With Infinity Values ([#2709](../../pull/2709))
* Scatter Plot: regression line: show r instead of k ([#2701](../../pull/2701))
* settings: Do not clear schema_only settings on close_context ([#2691](../../pull/2691))
* Statistics.unique: Fix Sparse Return Order For Negative Numbers ([#2572](../../pull/2572))
* Statistics.countnans/bincount: Fix NaN Counting, Consider Implicit Zeros ([#2698](../../pull/2698))
* MDS: No optimization when subset data ([#2675](../../pull/2675))
* Outliers widget no longer checks classes and doesn't crash on singular covariances matrices ([#2677](../../pull/2677))
* OWRank: Fix autocommit ([#2685](../../pull/2685))
* OWScatterPlot: Change output Feature to AttributeList ([#2689](../../pull/2689))
* OWSql does not save selected table/query ([#2659](../../pull/2659))
* Scatter Plot: Scatter Plot automatically sends selection ([#2649](../../pull/2649))
* Silhouette plot rendering ([#2656](../../pull/2656))
* Variable.make returns proxies ([#2667](../../pull/2667))
* owhierarchicalclustering: Fix performance on deselection ([#2670](../../pull/2670))
* Report Table: Make Table Headers Bold ([#2668](../../pull/2668))
* MDS: primitive metas, init_attr_values ([#2661](../../pull/2661))
* MDS: Primitive metas ([#2648](../../pull/2648))
* MDS: similar pairs and combos not cleared ([#2643](../../pull/2643))
* Scatter Plot: remove dead and commented code, tests ([#2627](../../pull/2627))


[3.6.0] - 2017-09-29
--------------------
##### Enhancements
* PythonScript: Multiple inputs ([#2506](../../pull/2506))
* DomainEditor: Add horizontal header ([#2579](../../pull/2579))
* Feature Constructor: Support additional functions () ([#2611](../../pull/2611))
* Miniconda installer: Install conda executable ([#2616](../../pull/2616))
* Datasets: New widget ([#2557](../../pull/2557))
* Neural Network widget ([#2553](../../pull/2553))

##### Bugfixes
* settings: Store settings version in the serialized defaults ([#2631](../../pull/2631))
* canvas/stackedwidget: Check if the new geometry is the same as the old ([#2636](../../pull/2636))
* OWRank: sort NaNs last; fix sort indicator ([#2618](../../pull/2618))
* Schema-only settings in components ([#2613](../../pull/2613))
* OWBaseLearner: Save learner name in workflow ([#2608](../../pull/2608))
* Saving of multiple selections in ScatterPlot ([#2598](../../pull/2598))
* OWBoxPlot: Faster selection ([#2595](../../pull/2595))
* preprocess.randomization: Do not use the same seed for X, Y, and meta ([#2603](../../pull/2603))
* Slow Rank ([#2494](../../pull/2494))
* setup: Increase required setuptools version ([#2602](../../pull/2602))
* Disable pyqtgraph's exit cleanup handler ([#2597](../../pull/2597))
* ScatterPlotGraph: fix labelling when there are missing data ([#2590](../../pull/2590))
* canvas: Fix link runtime state modeling ([#2591](../../pull/2591))
* Tree: Reintroduce preprocessors. ([#2566](../../pull/2566))
* canvas/preview: Fix workflow preview rendering ([#2586](../../pull/2586))
* Fix saving reports on Python 3.6 ([#2584](../../pull/2584))
* Fix failing report tests ([#2574](../../pull/2574))
* widgets/tests: Compatibility with Python 3.5.{0,1} ([#2575](../../pull/2575))


[3.5.0] - 2017-09-04
--------------------
##### Enhancements
* Proper calculation of distances ([#2454](../../pull/2454))
* OWFeatureConstructor: Add new functions from numpy ([#2410](../../pull/2410))
* Widget status bar ([#2464](../../pull/2464))
* Impute widget: Parallel execution in the background ([#2395](../../pull/2395))

##### Bugfixes
* Mosaic Display: subset data ([#2528](../../pull/2528))
* MDS: Optimize similar pairs graphics construction ([#2536](../../pull/2536))
* Error Reporting: read attached schema file as utf8 ([#2416](../../pull/2416))
* Another color palette when too many colors needed ([#2522](../../pull/2522))
* Widget: splitter sizes ([#2524](../../pull/2524))
* Silhouette Plot: another memory error ([#2521](../../pull/2521))
* Fix asynchronous widget tests ([#2520](../../pull/2520))
* Mosaic Vizrank: compute_attr_order is called every step ([#2484](../../pull/2484))
* widgets/model: Restore 'explicit' hint flag for 'Coefficients' output ([#2509](../../pull/2509))


[3.4.5] - 2017-07-27
--------------------
##### Enhancements
* OWMDS, OWLinearProjection: Save selection in workflow ([#2301](../../pull/2301))
* SQL: Save user name and password via credentials manager ([#2403](../../pull/2403))
* Canvas Annotations: Text markup editing ([#2422](../../pull/2422))
* New windows installer scripts ([#2338](../../pull/2338))

##### Bugfixes
* Tree: Fix min_samples_leaf check ([#2507](../../pull/2507))
* Tree: Support classification on sparse data ([#2430](../../pull/2430))
* Trees: Support regression on sparse data ([#2497](../../pull/2497))
* Trees: Fix predictions on sparse data ([#2496](../../pull/2496))
* Change Variable Icons: Discrete -> Categorical, Continuous -> Numeric ([#2477](../../pull/2477))
* Distributions: Show probabilities upon selection ([#2428](../../pull/2428))
* Manifold Learning: Handling out of memory error ([#2441](../../pull/2441))
* CN2 Rule Induction: Handling out of memory error ([#2397](../../pull/2397))
* Hierarchical Clustering: Explicit geometry transform ([#2465](../../pull/2465))
* Scatter Plot Graph: Legend symbols color ([#2487](../../pull/2487))
* Table: Fix printing data with sparse Y ([#2457](../../pull/2457))
* Ensure visible nodes after opening a workflow. ([#2490](../../pull/2490))
* Select Rows: Removing Unused Values for Discrete Variables in Sparse Data ([#2452](../../pull/2452))
* simple_tree.c: Fix mingw compiler compatibility ([#2479](../../pull/2479))
* Add-ons: Fix Installation of Official Add-ons Through Drag & Drop ([#2481](../../pull/2481))
* Mosaic: Clear when data is disconnected ([#2462](../../pull/2462))
* Create Class: Class name cannot be empty ([#2440](../../pull/2440))
* WidgetSignalsMixin: Fix input/output ordering for 'newstyle' signals ([#2469](../../pull/2469))
* Table: Update `ids` in `Table.__del__` ([#2470](../../pull/2470))
* Preprocess: Fix RemoveNaNClasses / Use existing HasClass ([#2450](../../pull/2450))
* SQL: Fixes for Custom SQL option ([#2456](../../pull/2456))
* OWColor: Fix propagating changes to the output ([#2379](../../pull/2379))
* Distances: Prevent inf numbers ([#2380](../../pull/2380))
* Test and Score: Show default columns ([#2437](../../pull/2437))
* Silhouette Plot: Now setting axis range properly ([#2377](../../pull/2377))
* Logistic Regression: Impute ([#2392](../../pull/2392))
* schemeedit: Clear edit focus before removing items ([#2427](../../pull/2427))
* Disable menu and mouse zoom in all evaluate's plotting widgets. ([#2429](../../pull/2429))
* canvas: Fix proposed connection scoring for dynamic signals ([#2431](../../pull/2431))
* ROC Analysis: Color support for more than 9 evaluation learners ([#2394](../../pull/2394))
* Scatter Plot: Two minor errors ([#2381](../../pull/2381))
* Feature Constructor: No fail when no values ([#2417](../../pull/2417))


[3.4.4] - 2017-06-16
--------------------
##### Enhancements
* SimpleTreeLearner: Release GIL & thread safety ([#2398](../../pull/2398))
* Improve support for HiDPI displays ([#2325](../../pull/2325))
* Add a tutorial section on responsive GUI ([#2318](../../pull/2318))
* Check if updates are available upon startup ([#2273](../../pull/2273))

##### Bugfixes
* Vizrank: interact with gui from main thread only ([#2389](../../pull/2389))
* Some preprocessors couldn not be pickled ([#2409](../../pull/2409))
* MDS: Support distances without domain information ([#2335](../../pull/2335))
* Paint Data: Fix crash on empty data ([#2399](../../pull/2399))
* Distributions: do not crash on empty data ([#2383](../../pull/2383))
* Update checker: LooseVersion does not handle str parts ([#2401](../../pull/2401))
* owpreproces: Stable order of continuizers ([#2400](../../pull/2400))
* owmanifoldlearning: Remove `n_jobs=-2` parameter ([#2371](../../pull/2371))
* Scatter Plot: features and no data ([#2384](../../pull/2384))
* tests: Fix test errors when running with numpy 1.13.0 ([#2396](../../pull/2396))
* OWColor: Use DiscreteVariable values for matching contexts ([#2376](../../pull/2376))
* Outliers: handling memory error ([#2374](../../pull/2374))
* score.FCBF: do not segfault on continuous variables w/ <0 values ([#2355](../../pull/2355))
* Rank widget supports Scorer inputs ([#2350](../../pull/2350))
* Silhouette Plot: handling memory error ([#2336](../../pull/2336))
* Distances: handling errors due to too large arrays ([#2315](../../pull/2315))
* Confusion Matrix: do not append extra column if empty ([#2386](../../pull/2386))


[3.4.3] - 2017-06-03
--------------------
##### Enhancements
* Venn diagram: Support sparse data ([#2334](../../pull/2334))
* PCA: Support sparse data ([#2313](../../pull/2313))
* Impute: Support sparse data ([#2357](../../pull/2357))
* Merge: Support sparse data ([#2305](../../pull/2305))
* Scatter Plot: Support sparse data ([#2152](../../pull/2152))
* Manifold: Support t-SNE on sparse data ([#2281](../../pull/2281))
* Mosaic: Selectable color variable ([#2133](../../pull/2133))
* Test & Score: Allow choosing columns ([#2257](../../pull/2257))
* Preprocess: Add all available methods to feature selection ([#2205](../../pull/2205))
* Scatter Plot: Support string metas labels ([#2360](../../pull/2360))

##### Bugfixes
* Fix and improve Precision, Recall, F1 ([#2369](../../pull/2369))
* Paint Data: Stores data in list and not np.array ([#2314](../../pull/2314))
* Paint Data: Save and load labels ([#2259](../../pull/2259))
* File: No domain or empty domain -> no data ([#2337](../../pull/2337))
* File: Support sparse data in Domain Editor ([#2245](../../pull/2245))
* File: Raise and handle Exc. when file bad pickle ([#2232](../../pull/2232))
* Test & Score: Fix migration of old settings ([#2254](../../pull/2254))
* Test & Score: Show correct error ([#2263](../../pull/2263))
* Test & Score: Instantly recognize new input ([#2247](../../pull/2247))
* Test & Score: Handling memory errors ([#2316](../../pull/2316))
* Tree Viewer: Check if there is selected class value ([#2224](../../pull/2224))
* CredentialManager: Handling password credentials error ([#2354](../../pull/2354))
* RowInstance: Fix sparse check ([#2362](../../pull/2362))
* Cross Validation: Cast fold number to string ([#2348](../../pull/2348))
* Silhouette Plot: Elide hover labels if labels are long ([#2278](../../pull/2278))
* Select Rows, Table: Filtering string values ([#2176](../../pull/2176))
* Report: Handle PermissionError when trying to save ([#2225](../../pull/2225))
* Continuize: Prevent crashing - column with equal and NaN values ([#2144](../../pull/2144))
* Add-ons: Handling ValueError due to connection problems ([#2239](../../pull/2239))
* Correspondence: Prevent crashing when cont attr has one value ([#2149](../../pull/2149))
* WebEngineView: Insert JS if loading already started ([#2230](../../pull/2230))
* Manifold Learning: handling numpy LinAlgError ([#2228](../../pull/2228))
* MDS: Fix widget update scheduling ([#2211](../../pull/2211))
* Settings: Handle permission errors when saving settings ([#2147](../../pull/2147))
* Map: Minor fixes ([#2356](../../pull/2356))


[3.4.2] - 2017-04-19
--------------------
##### Enhancements
* Nomogram: Support for sparse data ([#2197](../../pull/2197))
* Add PDF format to image exporters ([#2210](../../pull/2210))
* Reimplement macOS application (.app) build scripts ([#2217](../../pull/2217))
* Canvas: Use 'windowFilePath' to display display current filename instead of the title ([#2206](../../pull/2206))
* OWTestLearners: Cross validation by feature ([#2145](../../pull/2145))
* Pythagorean tree: Make border scale invariant ([#2141](../../pull/2141))

##### Bugfixes
* Scatterplot crashes when loaded from old workflow ([#2241](../../pull/2241))
* Error Report: URL changed ([#2220](../../pull/2220))
* Scatter Plot: update class density ([#2238](../../pull/2238))
* KMeans: should not crash when there is less data rows than k ([#2172](../../pull/2172))
* Edit Domain: Prevent duplicate variable names ([#2146](../../pull/2146))
* Scatter Plot: left margin (axis y) is now adapting ([#2200](../../pull/2200))
* Predictions widget: handle similar but different domains ([#2129](../../pull/2129))
* OWNomogram: Do not paint scene until the widget is not open ([#2202](../../pull/2202))
* Test & Score: crashing prevented when learner disconnects ([#2194](../../pull/2194))
* Widget Logistic Regression: can handle unused values ([#2116](../../pull/2116))
* stats: Open Corpus in OWDataTable after transposing it ([#2177](../../pull/2177))
* Rank: fixes creating Table with infinite numbers ([#2168](../../pull/2168))
* Add-ons: Problems with datetime parsing ([#2196](../../pull/2196))
* OWPredictions: Allow classification when data has no target column ([#2183](../../pull/2183))
* OWDataSampler: Fix typo boostrap to bootstrap ([#2195](../../pull/2195))
* All widgets are set to auto* when they are used for the first time ([#2136](../../pull/2136))
* Preprocess: enums containing function names changed ([#2151](../../pull/2151))
* Fitter: Fix used_vals and params not being set ([#2138](../../pull/2138))
* VizRankDialog: Stop computation when parent widget is deleted ([#2118](../../pull/2118))
* Distributions Report: Visualizations are now fitted ([#2130](../../pull/2130))
* Fitter: Change params uses default if None ([#2127](../../pull/2127))
* Fix invalid settings reuse in File widget  ([#2137](../../pull/2137))
* Scatter Plot - Prevent crash due to missing data ([#2122](../../pull/2122))
* Sieve Diagram: Using datasets with meta data ([#2098](../../pull/2098))


[3.4.1] - 2017-03-16
--------------------
##### Enhancements
* Scatterplot: Implement grouping of selections ([#2070](../../pull/2070))

##### Bugfixes
* Discover widgets when some dependencies are missing ([#2103](../../pull/2103))
* Select Rows: "is defined" fails ([#2087](../../pull/2087))
* report comments and OWFile reporting filename ([#1956](../../pull/1956))
* owcorrespondence: Handle variables with one value ([#2066](../../pull/2066))
* OWTreeViewer: Fix trees being displayed differently for same tree object ([#2067](../../pull/2067))
* Fitter: Properly delegate preprocessors ([#2093](../../pull/2093))


[3.4.0] - 2017-03-06
--------------------
##### Enhancements
* OWSGD: Output coefficients ([#1981](../../pull/1981))
* OWNomogram: Add a new widget ([#1936](../../pull/1936))
* OWRandomize: Add a new widget ([#1863](../../pull/1863))
* Map widget ([#1735](../../pull/1735))
* Table.transpose: Use heuristic to guess data type of attributes of attributes ([#1844](../../pull/1844))
* Create Class widget ([#1766](../../pull/1766))

##### Bugfixes
* Heatmap: Fix crash on data with empty columns ([#2057](../../pull/2057))
* ScatterPlot: Fix crash when coloring by column of unknowns ([#2061](../../pull/2061))
* owpreprocess: Handle columns with only NaN values ([#2064](../../pull/2064))
* File: Disallow changing string columns to datetime ([#2050](../../pull/2050))
* OWKMeans: Auto-commit fix and silhuette optimization ([#2073](../../pull/2073))
* OWDistributions: Fix binning of meta attributes ([#2068](../../pull/2068))
* SelectRows: Fix loading of conditions ([#2065](../../pull/2065))
* OWRandomize: New icon ([#2069](../../pull/2069))
* ZeroDivisionError owmosaic.py ([#2046](../../pull/2046))
* OWMosaic:  Fix crash for empty column ([#2006](../../pull/2006))
* Fitter: Fix infinite recursion in __getattr__ ([#1977](../../pull/1977))
* OWTreeGraph: Update node text when selecting target class ([#2045](../../pull/2045))
* Prevent PickleError (owfile.py) ([#2039](../../pull/2039))
* Fix Chi2 computation for variables with values with no instances ([#2031](../../pull/2031))
* OWDistanceMatrix: Remove quotes with string labels ([#2034](../../pull/2034))
* owheatmap: Prevent sliders to set Low >= High ([#2025](../../pull/2025))
* WebviewWidget: WebEngine Don't Grab Focus on setHtml ([#1983](../../pull/1983))
* OWFile: Show error msg when file doesn't exists ([#2024](../../pull/2024))
* Preprocess Widget: Continuize type error ([#1978](../../pull/1978))
* data/io.py Metadata file not saved anymore when it is empty ([#2002](../../pull/2002))
* Import from AnyQt instead from PyQt4 ([#2004](../../pull/2004))
* OWNomogram: Adjust scene rect ([#1982](../../pull/1982))
* owconcatenate: Fix domain intersection (remove duplicates) ([#1967](../../pull/1967))
* preprocess: Reset number_of_decimals after scaling ([#1914](../../pull/1914))
* Treeviewer sklearn tree compatibility ([#1870](../../pull/1870))
* OWSVR: Update learner when SVR type changes ([#1878](../../pull/1878))
* Tree widget binarization ([#1837](../../pull/1837))


[3.3.12] - 2017-02-14
--------------------
##### Bugfixes
* Highcharts: Fix freezing on Qt5 ([#2015](../../pull/2015))
* Handle KeyError Mosaic Display (owmosaic.py) ([#2014](../../pull/2014))
* Loading iris on C locale ([#1998](../../pull/1998))
* Handle KeyError Sieve Diagram widget (owsieve) when one row ([#2007](../../pull/2007))
* Test Learners: Fix AUC for selected single target class ([#1996](../../pull/1996))
* OWDataSampler: Fix 'Fixed proportion of data' option ([#1995](../../pull/1995))


[3.3.11] - 2017-02-03
--------------------
##### Enhancements
* Widget testing utilities ([#1939](../../pull/1939))

##### Bugfixes
* KMeans: Fix crashes when underlying algorithm fails ([#1974](../../pull/1974))
* owpaintdata: Adjust color model to input dataset ([#1988](../../pull/1988))
* scatterplot: Fix density image ([#1990](../../pull/1990))
* owpaintdata: Fix an error when the input dataset contains NaN ([#1972](../../pull/1972))
* Table: Ensure correct dtype in `_compute_distributions` ([#1969](../../pull/1969))
* Evaluation Results input validation ([#1954](../../pull/1954))
* owimpute: Fix editing of individual imputers ([#1966](../../pull/1966))
* gui: Trigger callback in SpinBoxWFocusOut only if value changed ([#1979](../../pull/1979))
* Python 3.6 compatibility ([#1963](../../pull/1963))
* File: Fix crash when last_path is None ([#1961](../../pull/1961))
* Paint Data: in-place output modification ([#1959](../../pull/1959))
* DataSampler: Fix crash when stratifying unbalanced datasets ([#1952](../../pull/1952))
* Table.__repr__: Fix for sparse data with < 5 instances ([#1951](../../pull/1951))
* Catch errors during learning in learner widgets ([#1949](../../pull/1949))
* OWMosaic: Fix crash for attribute with no values ([#1941](../../pull/1941))
* Impute: Fix crash when model-based imputation fails ([#1937](../../pull/1937))
* OWSieve: Fix crash for attribute with no values ([#1934](../../pull/1934))
* Tree: Fix crash when two attributes equal number of values ([#1928](../../pull/1928))
* Store changed variables in File widget ([#1805](../../pull/1805))


[3.3.10] - 2017-01-18
--------------------
##### Enhancements
* Input/output signal replacement declarations ([#1810](../../pull/1810))

##### Bugfixes
* MDS Widget: Handle NaN values for plot point styling ([#1931](../../pull/1931))
* OWPCA: Fix crash for dataset with no rows or no attributes ([#1915](../../pull/1915))
* OWMosaic: Discretize metas as well ([#1912](../../pull/1912))
* owfeaturecontructor: Fix an IndexError accessing exception's args ([#1905](../../pull/1905))
* owrank: Remove `super()` call from `migrate_settings` ([#1902](../../pull/1902))
* OWBoxPlot: Fix ordering of boxes ([#1900](../../pull/1900))
* canvas/readwrite: Fix byte literal serialization ([#1898](../../pull/1898))
* owpca: Handle the case of 0 total variance in the PCA solution ([#1897](../../pull/1897))
* Copy data attributes for annotated data set ([#1895](../../pull/1895))
* colorpalette: Fix AttributeError ([#1889](../../pull/1889))
* OWDistributions: Reset combos when data is removed ([#1887](../../pull/1887))
* Concatenate bugfix ([#1886](../../pull/1886))
* OWPredictions: Fix crash when opening report ([#1884](../../pull/1884))
* owsilhouetteplot: Fix TypeError when cluster column is an object array ([#1876](../../pull/1876))
* OWSave: Safer Check if Writer Support Sparse ([#1864](../../pull/1864))
* OWImageViewer: Fix selection with missing values ([#1861](../../pull/1861))
* owselectcolumns: Fix auto commit on any change ([#1859](../../pull/1859))
* Table.transpose: Keep metas array two dimensional when no attributes in domain ([#1855](../../pull/1855))
* Select Rows filter enum ([#1854](../../pull/1854))
* Scatter plot: don't crash on report without data ([#1840](../../pull/1840))
* Crash on ctrl-c/cmd-c in widgets without graphs ([#1827](../../pull/1827))
* Fix crash in listview if labels are changed before calling __setitem__ ([#1825](../../pull/1825))
* Scatterplot: Allow labelling by string attributes ([#1812](../../pull/1812))
* Fix copy to clipboard in "Data Table" widget ([#1808](../../pull/1808))
* TreeGraph: Compatibility with old schemas ([#1804](../../pull/1804))


[3.3.9] - 2016-12-02
--------------------
##### Enhancements
* OWTranspose: Add a new widget ([#1738](../../pull/1738))
* Add appveyor configuration ([#1693](../../pull/1693))
* Vizrank indicators and filters ([#1746](../../pull/1746))
* OWManifoldLearning: MDS - enable PCA initialization ([#1702](../../pull/1702))
* Add VizRank to Mosaic ([#1699](../../pull/1699))
* Setting migration ([#1724](../../pull/1724))
* widget: Allow subclasses to disable the default message bar widget ([#1543](../../pull/1543))
* Manifold Learning ([#1624](../../pull/1624))
* SQL Server support in SQL widget ([#1674](../../pull/1674))
* Visualize widgets: Output Annotated data and Fixups ([#1677](../../pull/1677))
* Add support for PyQt5 ([#1171](../../pull/1171))
* Simple benchmarking suite. ([#1510](../../pull/1510))
* Canvas: Always show the link dialog if the user holds Shift ([#1673](../../pull/1673))
* Scatterplot, HeatMap, TreeGraph, ConfusionMatrix and Unsupervised widgets: Output Flagged Data  ([#1655](../../pull/1655))
* CN2RuleViewer: Output sample of training data in absence of separate data ([#1667](../../pull/1667))
* Metadata for data files ([#1603](../../pull/1603))

##### Bugfixes
* owrank: Add migrate_settings ([#1797](../../pull/1797))
* owconfusionmatix: Add migrate_settings ([#1796](../../pull/1796))
* Improve ada boost widget ([#1787](../../pull/1787))
* OWBoxPlot: Fixups ([#1783](../../pull/1783))
* Filter: Fix FilterContinuous eq operator ([#1784](../../pull/1784))
* OWDistanceMatrix: attribute in context ([#1761](../../pull/1761))
* Hierarchical clustering: Make annotation a context setting ([#1748](../../pull/1748))
* Fix varius deprecation (and other) warnings ([#1774](../../pull/1774))
* Fix transformation for non primitive variables ([#1770](../../pull/1770))
* TimeVariable: don't crash Data Table when reloading and Visualize ... ([#1760](../../pull/1760))
* OWDistances: Mahalanobis wrong dimensions notification. ([#1762](../../pull/1762))
* Switch Sieve to DomainModel, which also fixes VizRank crash on meta attributes ([#1642](../../pull/1642))
* Confusion matrix: Map annotated data through row_indices, add probabi… ([#1720](../../pull/1720))
* OWLoadClassifier: Show message on unpickling error ([#1752](../../pull/1752))
* Silhouette Plot: Fixes ([#1747](../../pull/1747))
* canvas/toolgrid: Remove (unused) mouse press event tracking ([#1740](../../pull/1740))
* Box plot: Handle situation when quantiles can't be computed ([#1742](../../pull/1742))
* owfile: Hide apply button after resetting editor_model ([#1711](../../pull/1711))
* oweditdomain: Initialize `var` attribute ([#1731](../../pull/1731))
* FeatureConstructor: Fix crash when new variable is created without data ([#1733](../../pull/1733))
* owpythonscript: Fix QFileDialog.get{Save,Open}FileName usage ([#1726](../../pull/1726))
* DendrogramWidget: Prevent a zero division error ([#1725](../../pull/1725))
* owfile: Skip add_origin if no filename ([#1717](../../pull/1717))
* OWFile: Do not load large files automatically ([#1703](../../pull/1703))
* Do not show messages when data is removed ([#1706](../../pull/1706))
* Confusion Matrix: Show error on regression results ([#1709](../../pull/1709))
* Fix tests ([#1698](../../pull/1698))
* Scatter Plot: Fix a error when restoring from pre DomainModel workflows ([#1672](../../pull/1672))
* Tree Scorers: Change 'int64_t' to 'intp_t' for platform independence ([#1687](../../pull/1687))
* OWTable: Sort Continuous metas as floats; not strings ([#1678](../../pull/1678))
* Error Reporting: Temporary last open/save directory ([#1676](../../pull/1676))
* TableModel: Don't crash on empty sparse data ([#1675](../../pull/1675))
* Statistics.util.stats: Fix negative #nans for sparse ([#1659](../../pull/1659))
* MDS Widget: Fix zero length line, gray square bug ([#1670](../../pull/1670))
* Fix an error when using latest pyqtgraph develop snapshot ([#1662](../../pull/1662))
* OWHeatMap: Resend 'Selected Data' when settings change ([#1664](../../pull/1664))
* Fix pythagoras tree tooltip for regression trees ([#1660](../../pull/1660))
* OWConfusionMatrix: Output None when no data is selected ([#1653](../../pull/1653))
* OWBoxPlot: Reset widget's appearance when data is removed ([#1654](../../pull/1654))


[3.3.8] - 2016-10-11
--------------------
##### Enhancements
* CredentialManager: Store passwords in System Keyring Services ([#1641](../../pull/1641))
* Extend widget creation policy ([#1611](../../pull/1611))
* File widget improvements ([#1607](../../pull/1607))
* Remote reporting of unexpected errors ([#1558](../../pull/1558))
* OWRank: Widget improvements ([#1560](../../pull/1560))
* canvas: Indicate runtime state on links ([#1554](../../pull/1554))
* Rule induction (CN2) ([#1397](../../pull/1397))
* Upgrade OWSvm unittests ([#1499](../../pull/1499))
* Enable Ward clustering in Hierarchical clustering widget  ([#1515](../../pull/1515))
* PCA transformation speedup ([#1539](../../pull/1539))

##### Bugfixes
* owsql: Fix bug when using connection before established ([#1638](../../pull/1638))
* Scatterplot: Reintroduce sliders for size and opacity ([#1622](../../pull/1622))
* Reporting tabular in Data Table and Rank widgets ([#1573](../../pull/1573))
* BoxPlot crashes on variables with no known values (Fixes #1568) ([#1647](../../pull/1647))
* Canvas: Replace illegal file-name characters with _ when saving workf… ([#1644](../../pull/1644))
* OWScatterPlot: Fix progress bar percentages running over 100% ([#1645](../../pull/1645))
* OWFile: Report errors for incorrect file formats instead of crashing ([#1635](../../pull/1635))
* OWFeatureConstructor: Fix domain check for only meta data sets ([#1632](../../pull/1632))
* gui.lineEdit: Restore changed state tracking ([#1630](../../pull/1630))
* errorreporting: Fix an KeyError for a missing 'Widget Module' entry ([#1625](../../pull/1625))
* win-installer: Build scikit-learn in windows installer build script ([#1623](../../pull/1623))
* owlearnerwidget: Fix output initialization ([#1562](../../pull/1562))
* gui: Add a push button class adapted for variable width text ([#1614](../../pull/1614))
* :  Add `Explicit` flag to supplementary Table learner output ([#1617](../../pull/1617))
* Silhouette: Auto-commit on changing checkbox state ([#1606](../../pull/1606))
* Linear regression: Fix Elastic net; Fix Auto-apply buttons ([#1601](../../pull/1601))
* ROC Analysis - Fix roc averaging ([#1595](../../pull/1595))
* OWBaseLearner: Do not re-fit if name has changed ([#1580](../../pull/1580))
* Context attributes with metas in Sieve and Mosaic ([#1545](../../pull/1545))
* Variable: Fix Variable.copy for StringVariable and TimeVariable ([#1589](../../pull/1589))
* Stats: Fix counting of missing values for non-numeric data ([#1585](../../pull/1585))
* Load Classifier widget sends classifier on init ([#1584](../../pull/1584))
* Context settings ([#1577](../../pull/1577))
* Fixed svg function to return svg chart together with container div for highcharts ([#1541](../../pull/1541))
* Fix compatibility with Color widget ([#1552](../../pull/1552))
* Gini impurity: formula and docstring fixed. ([#1495](../../pull/1495))
* owimageviewer: Open local images directly ([#1550](../../pull/1550))
* Fix loading of datasets with paths in variable attributes ([#1549](../../pull/1549))
* Confusion matrix: fix selected_learner setting ([#1523](../../pull/1523))
* canvas/addons: Remove wrong/unnecessary proxy mapping ([#1533](../../pull/1533))
* Scatterplot: Score Plots crash if multiple attributes have the same score ([#1535](../../pull/1535))
* ScatterPlot: Score Plots window title changed to title case ([#1525](../../pull/1525))
* Predictions: column size hint ([#1514](../../pull/1514))


[3.3.7] - 2016-08-05
--------------------
##### Enhancements
* ImageViewer: Add a 'Preview' like window ([#1402](../../pull/1402))
* Pythagorean tree and Pythagorean forest widgets ([#1441](../../pull/1441))
* New workflow examples for the Welcome screen ([#1438](../../pull/1438))
* Test widgets on travis ([#1417](../../pull/1417))
* Save painted data to schema ([#1452](../../pull/1452))
* Welcome screen: New icons for welcome screen ([#1436](../../pull/1436))
* SqlTable: Automatically recognize date/time fields ([#1424](../../pull/1424))
* Proxy support for add-on installation ([#1379](../../pull/1379))
* Automatically create required SQL extensions ([#1395](../../pull/1395))
* Ranking for Sieve, refactoring of Sieve, Mosaic and VizRank ([#1382](../../pull/1382))
* Rank adopted for sparse data ([#1399](../../pull/1399))
* PCA: Add lines and labels showing the explained variance ([#1383](../../pull/1383))
* Implement copying graph to clipboard using Ctrl-C (Cmd-C) ([#1386](../../pull/1386))
* Parallelized cross validation & other evaluation methods ([#1004](../../pull/1004))
* Image viewer thumbnail size ([#1381](../../pull/1381))

##### Bugfixes
* Table names set by readers ([#1481](../../pull/1481))
* OWRandomForest: Fix, refactor and widget tests ([#1477](../../pull/1477))
* KNN: Fix crash when Mahanalobis metric is used ([#1475](../../pull/1475))
* Fix AdaBoost widgets and add some tests ([#1474](../../pull/1474))
* Table: Fix ensure_copy for sparse matrices ([#1456](../../pull/1456))
* statistics.utils: Fix stats for sparse when last column missing ([#1432](../../pull/1432))
* MDS and Distances widges fix ([#1435](../../pull/1435))
* OWBaseLearner: Learner name is changed on output when user changes it and auto apply selected  ([#1453](../../pull/1453))
* Stop advertising support for weights in LogisticRegression. ([#1448](../../pull/1448))
* OWScatterPlot: Fix information message reference ([#1440](../../pull/1440))
* Fix Tree preprocessor order. ([#1447](../../pull/1447))
* SqlTable: Cast to text for unknown and string types ([#1430](../../pull/1430))
* NaiveBayes: Handle degenerate cases ([#1442](../../pull/1442))
* OWBoxPlot: Show corresponding label when ploting discrete variable ([#1400](../../pull/1400))
* Lin and Log Regression: Prevent double commit ([#1401](../../pull/1401))
* KMeans: Silhouette score format precision fixed to integer ([#1434](../../pull/1434))
* Select Rows: skip undefined TimeVariable filters ([#1429](../../pull/1429))
* OWTestLearners: Fix reporting results table ([#1421](../../pull/1421))
* Scatterplot: Sends none in no instance selected ([#1428](../../pull/1428))
* PCA: Fix the variance spin. ([#1396](../../pull/1396))
* overlay: Auto disconnect when the overlay widget is deleted ([#1412](../../pull/1412))
* PaintData: Send None instead of empty table when the plot is empty ([#1425](../../pull/1425))
* Rand Forest Class: Min sample replaces max leaf nodes ([#1403](../../pull/1403))
* SelectRows: Index attrs only by visible in set_data ([#1398](../../pull/1398))
* File: Stores filenames to image attributes ([#1393](../../pull/1393))
* Fix a logging error on windows ([#1390](../../pull/1390))
* OWLearnerWidget: Don't crash when training data contains no features  ([#1389](../../pull/1389))
* TimeVariable: fix repr rounding and repr for nan ([#1387](../../pull/1387))


[3.3.6] - 2016-06-24
--------------------
##### Enhancements
* Automatically discretize continuous variables in Sieve (#1372)
* Nicer reporting of tabular data (e.g. Confusion Matrix) (#1309)
* Match only at the beginning of word in quickMenu search (#1363)
* Univar regression coefficients (#1186)
* Add Mahalanobis distance (#1355)
* Move Data Table higher in the 'suggested widgets' list (#1346)
* Support Distances on Sparse Data (#1345)
* Add auto apply to Test & Score (#1344)
* Support unix timestamps in TimeVariable (#1335)
* Skip Hidden Attributes in SelectRows (#1324)
* Preprocessor widget: add random feature selection (#1112)
* Sort list of available add-ons (#1305)

##### Bugfixes
* Fix and improve handling of input data in PaintData (#1342)
* Mosaic: Output original data, not discretized (#1371)
* Fix report for OWLinearProjection (#1360, #1361)
* Fix auto-apply in SelectColumns (#1353)
* Fix a RuntimeError in ImageViewer when clearing the scene (#1357)
* Use https to query pypi API (#1343)
* OWBaseLearner: Add name attribute to learner (#1340)
* Pass data attributes after preprocess. (#1333)
* Better support for sparse data (#1306)
* gui.auto_commit: Fix crash when caller gives  argument (#1325)
* Fix image viewer runtime error (#1322)
* Better selection indicators in canvas (#1308)
* Open correct help file for add-on widgets (#1307)


[3.3.5] - 2016-06-01
--------------------
##### Bugfixes
* Revert hack that caused missing icons in osx build
* Fix installation in environments without numpy installed (#1291)
* Allow running of library test when PyQt is not available (#1289)


[3.3.4] - 2016-05-27
--------------------
##### Enhancements
* Install add-on by dragging zip/tgz/wheel onto the addons dialog (#1269)
* Added missing reports (#1270, #1271, #1272, #1273, #1279)
* Add auto-apply checkboxes to learner dialogs (#1263)
* Sort numeric values as numbers in Table (#1255)
* Open dragged files in OWFile (#1176)
* Support context cloning in FeatureConstructor and SelectRows (#1196)

##### Bugfixes
* Depend on scikit-learn>=0.17 (#1277)
* Fix installation problem on windows (#1278)
* Fix crash in K-means when silhouette cannot be computed (#1247)
* Fix crash in Distributions on empty data (#1246)
* Reset outputs in MergeData (#1240)
* Compute distances between constructed Instances (#1242)
* Fix links in Changelog (#1244)


[3.3.3] - 2016-05-03
--------------------
##### Bugfixes
* Revert installation of desktop launcher on Linux (#1218)
* Fix a crash when learner is connected to file (#1220)


[3.3.2] - 2016-04-22
--------------------
##### Enhancements
* New preprocessors ReliefF and FCBF (#1133)
* New feature scorers ANOVA, Chi2 and Univariate Linear Regression (#1125)
* Mosaic plot can display numeric attributes (#1165)
* Sheet selection for excel files in File widget (#1164)
* Check code quality with pylint on Travis (#1121)
* Improve PyQt5 forward compatibility (#1029)
* Include default datasets in File widget (#1174)
* Install desktop launcher on linux (#1205)

##### Bugfixes
* Fix a bug in nested transformation of instance (#1192)
* Fix vizrank's crash when pair had no valid data (#1189)
* Save Graph doesn't save axes (#1134)
* Open included tutorials with correct dataset (#1169)
* Disable bsp index on the main canvas scene (#1168, #1173)
* Fix FeatureConstructor crash with Python 3.5 (#1157)
* Include feature names in Rank Widget report (#1022)
* Decrease memory consumption in PCA (#1150)
* Fix dragging of treshold in PCA widget (#1051)
* Save TimeVariable in ISO 8601 format (#1145)
* Allow use of feature metadata in MDS (#1130)
* OWSelectColumns: fix drag and drop for features with Unicode names (#1144)
* Warn when setting values are not present on instance (#1139)
* Fix printing of Table with a TimeVariable (#1141)
* Fix Test & Score widget report (#1138)


[3.3.1] - 2016-03-24
--------------------
##### Enhancements
* Rank widget outputs scores
* SGD Regression widget: Fixed layout and added reporting

##### Bugfixes
* Windows installer: update pip on target system if required

[3.3] - 2016-03-18
------------------
##### Enhancements
*  Changed layout of File widget
*  Distance matrix widget
*  Meta attributes in Sieve and Mosaic
*  New type of variable: Time variable
*  Report for Distance Transformation widget
*  Report for Linear Regression widget
*  Report for Univariate Regression (fixes #1080)
*  score.FCBF: a Fast Correlation-Based Filter for feature selection
*  Sieve enhancements
*  Silhouette Plot widget
*  Venn Diagram: Add option to output unique/all instances.
*  Widgets for saving and loading distances

##### Bugfixes
*  breast-cancer.tab: change type of tumor-size column (fixes #1065)
*  Color: Do not resize columns in continuous table to contents (fixes #1055)
*  Exporting graphs to dot format
*  OWDistributions: Do not remove constant attribute and do not draw if there is no data
*  OWRank: Give name to Scores output table
*  OWRank no longer crashes when additional learners are available
*  ReliefF: Support for missing target values
*  Report: Fix crash on reporting tables
*  RF without pruning by default

##### Documentation
* Update build/install/contributing READMEs
* Update documentation in widget.rst

[3.2] - 2016-02-12
------------------
* Finalized Orange 3.2, switched to stable(r) release cycles

[0.1] - 1996-10-10
----------------
* Initial version based on Python 1.5.2 and Qt 2.3


[next]: https://github.com/biolab/orange3/compare/3.13.0...HEAD
[3.13.0]: https://github.com/biolab/orange3/compare/3.12.0...3.13.0
[3.12.0]: https://github.com/biolab/orange3/compare/3.11.0...3.12.0
[3.11.0]: https://github.com/biolab/orange3/compare/3.10.0...3.11.0
[3.10.0]: https://github.com/biolab/orange3/compare/3.9.1...3.10.0
[3.9.1]: https://github.com/biolab/orange3/compare/3.9.0...3.9.1
[3.9.0]: https://github.com/biolab/orange3/compare/3.8.0...3.9.0
[3.8.0]: https://github.com/biolab/orange3/compare/3.7.1...3.8.0
[3.7.1]: https://github.com/biolab/orange3/compare/3.7.0...3.7.1
[3.7.0]: https://github.com/biolab/orange3/compare/3.6.0...3.7.0
[3.6.0]: https://github.com/biolab/orange3/compare/3.5.0...3.6.0
[3.5.0]: https://github.com/biolab/orange3/compare/3.4.5...3.5
[3.4.5]: https://github.com/biolab/orange3/compare/3.4.4...3.4.5
[3.4.4]: https://github.com/biolab/orange3/compare/3.4.3...3.4.4
[3.4.3]: https://github.com/biolab/orange3/compare/3.4.2...3.4.3
[3.4.2]: https://github.com/biolab/orange3/compare/3.4.1...3.4.2
[3.4.1]: https://github.com/biolab/orange3/compare/3.4.0...3.4.1
[3.4.0]: https://github.com/biolab/orange3/compare/3.3.12...3.4.0
[3.3.12]: https://github.com/biolab/orange3/compare/3.3.11...3.3.12
[3.3.11]: https://github.com/biolab/orange3/compare/3.3.10...3.3.11
[3.3.10]: https://github.com/biolab/orange3/compare/3.3.9...3.3.10
[3.3.9]: https://github.com/biolab/orange3/compare/3.3.8...3.3.9
[3.3.8]: https://github.com/biolab/orange3/compare/3.3.7...3.3.8
[3.3.7]: https://github.com/biolab/orange3/compare/3.3.6...3.3.7
[3.3.6]: https://github.com/biolab/orange3/compare/3.3.5...3.3.6
[3.3.5]: https://github.com/biolab/orange3/compare/3.3.4...3.3.5
[3.3.4]: https://github.com/biolab/orange3/compare/3.3.3...3.3.4
[3.3.3]: https://github.com/biolab/orange3/compare/3.3.2...3.3.3
[3.3.2]: https://github.com/biolab/orange3/compare/3.3.1...3.3.2
[3.3.1]: https://github.com/biolab/orange3/compare/3.3...3.3.1
[3.3]: https://github.com/biolab/orange3/compare/3.2...3.3
[3.2]: https://github.com/biolab/orange3/compare/3.1...3.2
[0.1]: https://web.archive.org/web/20040904090723/http://www.ailab.si/orange/
