Change Log
==========

[next] - TBA
------------
* ...


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


[next]: https://github.com/biolab/orange3/compare/3.4.4...HEAD
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
