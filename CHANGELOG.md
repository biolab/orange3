Change Log
==========

[next] - TBA
------------
* ...

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


[next]: https://github.com/biolab/orange3/compare/3.3.5...HEAD
[3.3.5]: https://github.com/biolab/orange3/compare/3.3.4...3.3.5
[3.3.4]: https://github.com/biolab/orange3/compare/3.3.3...3.3.4
[3.3.3]: https://github.com/biolab/orange3/compare/3.3.2...3.3.3
[3.3.2]: https://github.com/biolab/orange3/compare/3.3.1...3.3.2
[3.3.1]: https://github.com/biolab/orange3/compare/3.3...3.3.1
[3.3]: https://github.com/biolab/orange3/compare/3.2...3.3
[3.2]: https://github.com/biolab/orange3/compare/3.1...3.2
[0.1]: https://web.archive.org/web/20040904090723/http://www.ailab.si/orange/
