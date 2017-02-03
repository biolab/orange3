Change Log
==========

[next] - TBA
------------
* ...


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


[next]: https://github.com/biolab/orange3/compare/3.3.11...HEAD
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
