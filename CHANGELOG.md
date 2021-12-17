Change Log
==========

[next] - TBA
------------

[3.31.0] - 2021-12-17
--------------------
##### Enhancements
* oweditdomain: Indicate variables in error state ([#5732](../../pull/5732))
* Scatterplot: Use opacity for contrast ([#5684](../../pull/5684))
* Feature Constructor: Evaluate categorical variables to strings ([#5637](../../pull/5637))
* New widget: Group By  ([#5541](../../pull/5541))
* Table lock: tests run with tables that are read-only by default ([#5381](../../pull/5381))
* config: sort example workflows ([#5600](../../pull/5600))

##### Bugfixes
* Paint Data: Fix ClearTool's issued commands ([#5718](../../pull/5718))
* pandas_compat: fix table_from_frames for "normal" dataframe ([#5652](../../pull/5652))
* pandas_compat: do not parse column of numbers (object dtype) to datetime ([#5681](../../pull/5681))
* HeatMap: Color gradient center value edit ([#5647](../../pull/5647))
* Distance Matrix: Fix crash on numeric meta vars as labels ([#5664](../../pull/5664))
* Fix running OWS with widgets with WebView in Orange.canvas.run ([#5657](../../pull/5657))
* main: Fix `--clear-widget-settings` parameter declaration ([#5619](../../pull/5619))


[3.30.2] - 2021-10-27
--------------------
##### Bugfixes
* Projections: Fix color density for continuous color palettes ([#5665](../../pull/5665))
* Fixes for scikit-learn 1.0 ([#5608](../../pull/5608))
* table_from_grames: fix indices parsing ([#5620](../../pull/5620))
* Fix overflow in bin calculations for time variables.  ([#5667](../../pull/5667))
* Variable: fix timezone when parsing time variable ([#5617](../../pull/5617))
* Require widget-base 4.15.1 and canvas-core 0.1.23 to fix some bugs/crashes


[3.30.1] - 2021-09-24
--------------------
##### Bugfixes
* OWTable: fix select whole rows regression ([#5605](../../pull/5605))


[3.30.0] - 2021-09-22
--------------------
##### Enhancements
* OWPythonScript: Better text editor ([#5208](../../pull/5208))
* PCA: Output variance of components ([#5513](../../pull/5513))
* Curve Fit: New widget ([#5481](../../pull/5481))
* Hierarchical Clustering: Annotate variables with clusters ([#5514](../../pull/5514))
* Create nodes on canvas drag/drop ([#5031](../../pull/5031))

##### Bugfixes
* setup.py: do not overwrite conda's PyQt5 ([#5593](../../pull/5593))
* Use explicit ordered multiple inputs ([#4860](../../pull/4860))
* Prevent crash when saving in unsupported format ([#5560](../../pull/5560))
* owrocanalysis: Fix test for non empty points array ([#5571](../../pull/5571))
* pandas_compat: fix conversion of datetime series ([#5547](../../pull/5547))
* Fix deepcopy and pickle for classes derived from `np.ndarray` ([#5536](../../pull/5536))
* owheatmap: Fix assertion error when restoring selection ([#5517](../../pull/5517))
* Pivot: Handle empty data, metas only ([#5527](../../pull/5527))
* table_to_frame - handle numeric columns with dtype=object ([#5474](../../pull/5474))
* listfilter: Bypass QListView.dropEvent ([#5477](../../pull/5477))


[3.29.3] - 2021-06-09
--------------------
##### Bugfixes
* Create Class: fix incorrect value assignment


[3.29.2] - 2021-06-08
--------------------
##### Bugfixes
* Bump orange-canvas-core minimum version requirement ([#5472](../../pull/5472))
* owpca: fix component selection when dragging selection line ([#5469](../../pull/5469))
* Save File when workflow basedir is an empty string ([#5459](../../pull/5459))


[3.29.1] - 2021-05-31
--------------------


[3.29.0] - 2021-05-28
--------------------
##### Enhancements
* Scatter plot: Bring discrete attributes functionality back ([#5440](../../pull/5440))
* Caching data delegate ([#5296](../../pull/5296))
* textimport: Mark encoding errors in the preview ([#5438](../../pull/5438))
* DBSCAN: Optional normalization ([#5428](../../pull/5428))
* Automated and better summaries ([#5308](../../pull/5308))
* Transpose: Offload work onto separate thread, remove redundant instance ([#5314](../../pull/5314))
* Domain transformations in batches for less memory use ([#5218](../../pull/5218))
* Feature Statistics: Add median ([#5325](../../pull/5325))
* New widget: Aggregate Columns ([#5256](../../pull/5256))

##### Bugfixes
* Outlier detection: keep instance ids, make thread safe ([#5427](../../pull/5427))
* UrlReader: Support urls with special characters ([#5412](../../pull/5412))
* Speed-up slow table_to_frame ([#5413](../../pull/5413))
* Line Plot: Single instance input fix ([#5408](../../pull/5408))
* Pivot: Assign dataset name to output tables ([#5404](../../pull/5404))
* setup.py: Exclude benchmark directory from the install ([#5392](../../pull/5392))
* Errors when converting negative timestamps on Windows ([#5388](../../pull/5388))
* Nomogram: Retain original compute_value ([#5382](../../pull/5382))
* Radviz VizRank: Implement on_selection_changed ([#5338](../../pull/5338))


[3.28.0] - 2021-03-05
--------------------
##### Enhancements
* Bar Plot: Improve "Group by" visualization ([#5301](../../pull/5301))
* Violin Plot: New widget ([#5252](../../pull/5252))
* Test and Score: Copy selected rows to clipboard ([#5203](../../pull/5203))
* Projections: Allow transparent subset ([#5141](../../pull/5141))
* Gradient Boosting: New widget ([#5160](../../pull/5160))
* Impute: Allow setting a default value for all numeric and time variables ([#5102](../../pull/5102))
* Distribution: Show equal bar widths on unique-valued bins ([#5139](../../pull/5139))
* Implement proper Lift curve; keep Cumulative gains as an option ([#5075](../../pull/5075))
* Create Instance: New widget ([#5033](../../pull/5033))
* Add add_column and concatenate methods to Table ([#5251](../../pull/5251))

##### Bugfixes
* Calibration model: Work with numpy data ([#5159](../../pull/5159))
* Rank: Switch to manual selection on deselect ([#5271](../../pull/5271))
* Create Class: multiple patterns for a class value ([#5283](../../pull/5283))
* Test and Score: Fix stratification warnings ([#5281](../../pull/5281))
* Predictions: Fix crash when clicking on empty left area ([#5222](../../pull/5222))
* Distribution: vectorize variance, speeds up normalization ([#5230](../../pull/5230))
* owimpute: Make `default_numeric` locale independant ([#5209](../../pull/5209))
* Pivot: Display time variable in time format ([#5212](../../pull/5212))
* OWScatterPlotBase: Ignore 'Other' when showing color regions ([#5214](../../pull/5214))
* Fix performance regression in scatterplot ([#5206](../../pull/5206))
* Pivot: Fix table for categorical variables ([#5193](../../pull/5193))
* Distance Matrix: Fix freeze with large selections ([#5176](../../pull/5176))
* Xls reader: Error as missing value ([#5192](../../pull/5192))
* owdataset: Do not capture self in closure ([#5198](../../pull/5198))
* ROC shows all points, including the last ([#5138](../../pull/5138))
* Pivot: Output date for group by table ([#5202](../../pull/5202))
* Enable classification tests ([#5168](../../pull/5168))
* Data Table: Fix freeze with large selections ([#5164](../../pull/5164))
* OWImpute: Preserve default method setting ([#5181](../../pull/5181))
* AdaBoost: Set maximum number of estimators to 10000 ([#5165](../../pull/5165))
* Feature Statistics: Error in time variable display ([#5152](../../pull/5152))
* owpythonscript: Use signal id as is ([#5147](../../pull/5147))
* SqlTable use empty string instead of None for StringVariable ([#5120](../../pull/5120))
* Pivot: Handle big dataset ([#5104](../../pull/5104))
* SQL: Fix the issue with database collation setting when retrieving column values ([#5089](../../pull/5089))
* impute: Remove class vars from input data for ReplaceUnknownsModel ([#5083](../../pull/5083))
* Edit Domain: Preserve renames on categories merge ([#5072](../../pull/5072))
* CSV File Import: sort discrete values naturally ([#5041](../../pull/5041))


[3.27.1] - 2020-10-23
--------------------
##### Bugfixes
* customizableplot.available_font_families: Fix for non-existent default family ([#5037](../../pull/5037))
* Raise canvas-core version to fix some problems with Qt 5.9 ([#5045](../../pull/5045))


[3.27.0] - 2020-10-09
--------------------
##### Enhancements
* Table: Re-add info box about data properties ([#5011](../../pull/5011))
* Rank widget computation in a separate thread ([#4908](../../pull/4908))
* Neighbors: improve exclusion of references, checkbox to (un)limit output data ([#4997](../../pull/4997))
* Bar Plot: New widget ([#4923](../../pull/4923))
* Replace listViews with searchable listViews ([#4924](../../pull/4924))
* Add an option to set intercept in linear regression to 0 ([#4958](../../pull/4958))
* CSV File Import: Add support for explicit workflow relative paths ([#4872](../../pull/4872))
* OWColor: Saving and loading color schemata ([#4977](../../pull/4977))
* Distributions: Add sorting by category size ([#4959](../../pull/4959))
* Proxy support for embeddings ([#4953](../../pull/4953))
* Discretize: Manual cut points entry ([#4929](../../pull/4929))
* Predictions: Allow selecting a subset of rows ([#4871](../../pull/4871))
* Projection plots: Customize labels ([#4828](../../pull/4828))
* get_unique_names: Handle more independent names ([#4866](../../pull/4866))
* Edit Domain: Add option to unlink variable from source variable ([#4863](../../pull/4863))
* Louvain Clustering: Add cosine similarity ([#4864](../../pull/4864))

##### Bugfixes
* Fix line plot's send_report ([#5018](../../pull/5018))
* Scatter Plot: fix unzoom with density plot ([#5004](../../pull/5004))
* ownomogram: Fix wrapped C++ obj error ([#5005](../../pull/5005))
* Fix slicing in from_table ([#4963](../../pull/4963))
* Edit Domain: Multiple item rename/merge ([#4949](../../pull/4949))
* ProjectionWidgetMixinTests: set shape attribute only when discrete var available ([#4946](../../pull/4946))
* Fix variables equality and hashes  ([#4957](../../pull/4957))
* Fix wrong assert in heatmap ([#4955](../../pull/4955))
* Edit Domain (and perhaps other widgets) could cause missing data later in the workflow ([#4922](../../pull/4922))
* OWScatterPlotBase: Reset view before labels update ([#4907](../../pull/4907))
* owcsvimport: Fix a type error in _open for zip archive ([#4921](../../pull/4921))
* Line Plot: Reset axis ticks on data change ([#4873](../../pull/4873))
* MDS: Move lines when points are jittered ([#4920](../../pull/4920))
* owcsvimport: Handle decimal and thousand separator ([#4915](../../pull/4915))
* Select Rows: fix fail when time variable in metas ([#4912](../../pull/4912))
* Re-add TupleList to DiscreteVariable ([#4879](../../pull/4879))
* concurrent: Use a workaround for QObject wrapper deletion ([#4635](../../pull/4635))
* normalize: Adjust number_of_decimals after scaling ([#4779](../../pull/4779))


[3.26.0] - 2020-06-12
--------------------
##### Enhancements
* main: Log to main window output view ([#4842](../../pull/4842))
* Feature statistics report ([#4812](../../pull/4812))
* Distributions: change Histogram Data output ([#4832](../../pull/4832))
* owtable: output sorted data ([#4644](../../pull/4644))
* Add an option to Concatenate to merge columns with different formulae ([#4831](../../pull/4831))
* CSV Import: guess data types ([#4838](../../pull/4838))
* HeatMap: Allow setting the center when using diverging palettes ([#4809](../../pull/4809))
* Heatmap: Split columns, Column annotations ([#4703](../../pull/4703))
* Sort values naturally when reading files ([#4793](../../pull/4793))
* Color widget: Add reset button ([#4718](../../pull/4718))
* Gradient selection/parameters widget ([#4596](../../pull/4596))
* Select Rows: Allow partial context matches ([#4740](../../pull/4740))
* Edit Domain: Add an option to change the output table name ([#4722](../../pull/4722))
* ApplyDomain: data info displayed in the status bar ([#4611](../../pull/4611))
* BoxPlot: data info displayed in the status bar ([#4626](../../pull/4626))
* LinePlot: data info displayed in the status bar ([#4633](../../pull/4633))
* MosaicDisplay: data info displayed in the status bar ([#4634](../../pull/4634))
* CreateClass: data info displayed in the status bar ([#4625](../../pull/4625))

##### Bugfixes
* Variable: Fix cases when equal variables had different hashes ([#4843](../../pull/4843))
* OWBoxPlot: Fix wrong labels position and ordering for values with no items ([#4829](../../pull/4829))
* Select Rows: Fix saving meta variables in context ([#4830](../../pull/4830))
* Select Columns: Fix attributes sorting ([#4827](../../pull/4827))
* Fix and update Softmax regression learner ([#4767](../../pull/4767))
* PCA: fix subsets with the "Data" output ([#4811](../../pull/4811))
* OWContinuize: Fix treatment of continuous features. ([#4806](../../pull/4806))
* Select Rows: Fix incorrectly stored values in settings ([#4798](../../pull/4798))
* Fix colors for discrete variables with >256 values ([#4803](../../pull/4803))
* Unique domain checks ([#4760](../../pull/4760))
* owheatmap: Use 'is' instead of 'eq' for column id comparison ([#4788](../../pull/4788))
* BoxPlot: Fix invalid data range ([#4769](../../pull/4769))
* graphicstextlist: Fix size/spacing adjustment for single item ([#4777](../../pull/4777))
* Feature Statistics: Fix wrong or even crashing selections ([#4741](../../pull/4741))
* UrlReader: shorten TempFile extension ([#4747](../../pull/4747))
* Embedder: catch machine id setting type error ([#4675](../../pull/4675))
* relief: Fix contingency (de)allocation ([#4745](../../pull/4745))
* Test and Score: Improve data errors ([#4738](../../pull/4738))
* PythagoreanTree: Fix crushing when number of classes decreases ([#4743](../../pull/4743))
* Fix report in Predictions ([#4709](../../pull/4709))
* owheatmap: Handle all N/A column for color annotation ([#4742](../../pull/4742))
* Distributions widget's legend: Remove the square from sigma in normal and Rayleigh ([#4739](../../pull/4739))
* Several fixes in learners/models ([#4655](../../pull/4655))
* Heatmap: Split by missing values ([#4686](../../pull/4686))
* Owpaintdata, owpivot: ensure unique domain ([#4578](../../pull/4578))
* Pythagorantrees/forests: change context handler ([#4656](../../pull/4656))
* Color: Fix renaming of variables ([#4669](../../pull/4669))
* Heatmap: Sticky footer ([#4610](../../pull/4610))
* SOM: fix colors for numeric variables ([#4660](../../pull/4660))
* Fixes for deprecations in 3.26, and for changed behaviour of file dialog ([#4643](../../pull/4643))


[3.25.1] - 2020-05-22
--------------------
##### Bugfixes
* Fix compatibility with scikit-learn 0.23 ([#4768](../../pull/4768))


[3.25.0] - 2020-04-10
--------------------
##### Enhancements
* Searchable combo boxes in all evaluate widgets ([#4564](../../pull/4564))
* Searchable combo boxes in all visualize widgets ([#4563](../../pull/4563))
* Searchable combo boxes in all unsupervised widgets ([#4565](../../pull/4565))
* Projections keep colors after merging values ([#4577](../../pull/4577))
* Distance Map: Add a color map legend ([#4593](../../pull/4593))
* Orange.misc.environ config ([#4576](../../pull/4576))
* Searchable combo boxes in all data widgets ([#4562](../../pull/4562))
* Fix printing values with too few decimals ([#4575](../../pull/4575))
* Scatter Plot: Replace combo box with search combo box ([#4447](../../pull/4447))
* Save widgets: Store paths relative to workflow directory ([#4532](../../pull/4532))
* Edit Domain: Option to merge less frequent values ([#4477](../../pull/4477))
* Load Model: Use paths relative to workflow file ([#4534](../../pull/4534))
* Ignore missing values in density plots of projections ([#4525](../../pull/4525))
* Impose a sensible z-order to points in projections  ([#4504](../../pull/4504))
* Use Github actions as a new CI system. ([#4482](../../pull/4482))
* Testing with Tox ([#4481](../../pull/4481))
* Add row side color annotations ([#4443](../../pull/4443))
* Outliers: Offload work onto separate thread ([#4412](../../pull/4412))
* OWScatterPlot: axis displays time specific labels for time variable ([#4434](../../pull/4434))
* Predictions: Update splitter on resize ([#4433](../../pull/4433))
* Import openTSNE lazily for faster loading of Orange ([#4424](../../pull/4424))
* Silhouette Plot: Always output scores ([#4423](../../pull/4423))
* Heatmap: Tighter layout ([#4390](../../pull/4390))
* Reorganize continuous palettes ([#4305](../../pull/4305))
* Outliers: Save model into compute_value ([#4372](../../pull/4372))
* Allow concurrent transformation of tables into new domains ([#4363](../../pull/4363))
* Test & Score: Add comparison of models ([#4261](../../pull/4261))
* Outliers: Widget upgrade ([#4338](../../pull/4338))
* Concatenate: data info displayed in the status bar ([#4617](../../pull/4617))
* Distributions: data info displayed in the status bar ([#4627](../../pull/4627))
* MergeData: data info displayed in the status bar ([#4592](../../pull/4592))
* Neighbors: data info displayed in the status bar ([#4612](../../pull/4612))
* SelectByDataIndex: data info displayed in the status bar ([#4595](../../pull/4595))
* OWOutliers: Data info displayed in the status bar ([#4547](../../pull/4547))
* OWDataProjectionWidget: Upgrade status bar data info ([#4544](../../pull/4544))
* Heatmap: Restore ability to cluster larger datasets ([#4290](../../pull/4290))
* OWImpute: Data info displayed in the status bar ([#4499](../../pull/4499))
* OWFile: Data info displayed in the status bar ([#4506](../../pull/4506))
* OWCSVImport: Data info displayed in the status bar ([#4509](../../pull/4509))
* OWDatasets: Data info displayed in the status bar ([#4512](../../pull/4512))
* OWDataInfo: Data info displayed in the status bar ([#4513](../../pull/4513))
* OWSave: Data info displayed in the status bar ([#4505](../../pull/4505))
* OWPurgeDomain: Data info displayed in the status bar ([#4502](../../pull/4502))
* OWColor: Data info displayed in the status bar ([#4501](../../pull/4501))
* OWRandomize: Data info displayed in the status bar ([#4498](../../pull/4498))
* OWPivotTable: Data info displayed in the status bar ([#4472](../../pull/4472))
* OWContinuize: Data info displayed in the status bar ([#4494](../../pull/4494))
* OWFeatureConstructor: Data info displayed in the status bar ([#4496](../../pull/4496))
* OWSelectRows: Data info displayed in the status bar ([#4471](../../pull/4471))
* OWDiscretize: Data info displayed in the status bar ([#4495](../../pull/4495))
* OWDataSampler: Data info displayed in the status bar ([#4492](../../pull/4492))
* OWRank: Data info displayed in the status bar ([#4473](../../pull/4473))
* OWContinuize: Provide the same options as in Preprocess/Normalize ([#4466](../../pull/4466))
* OWCorrelations: Data info displayed in the status bar ([#4455](../../pull/4455))
* OWEditDomain: Data info displayed in the status bar ([#4456](../../pull/4456))
* OWSelectColumns: Data info displayed in the status bar ([#4454](../../pull/4454))
* OWTranspose: Data info displayed in the status bar ([#4413](../../pull/4413))
* OWFeatureStatistics: data info displayed in the status bar ([#4409](../../pull/4409))
* OWPreporcess: Data info displayed in the status bar ([#4414](../../pull/4414))

##### Bugfixes
* Edit Domain: Fix merge values when missing data ([#4636](../../pull/4636))
* Table: Send correct output when switching between tabs ([#4619](../../pull/4619))
* Give created QGraphicsScenes a parent ([#4352](../../pull/4352))
* Fix dimensionality of probabilities from values ([#4629](../../pull/4629))
* PyTableModel: Allow wrapping empty lists ([#4631](../../pull/4631))
* Edit Domain: Improve Text/Categorical to Time conversion ([#4601](../../pull/4601))
* colorpalettes: fix BinnedContinuousPalette color assignments ([#4609](../../pull/4609))
* Classification models output correct shapes ([#4602](../../pull/4602))
* owtestandscore: Add cancelled tasks to dispose queue ([#4615](../../pull/4615))
* Nomogram: Fix crash on Python 3.8 ([#4591](../../pull/4591))
* K-means slowness ([#4541](../../pull/4541))
* Use new access token in cleanup workflow ([#4590](../../pull/4590))
* Detect duplicate names of variables in projections. ([#4550](../../pull/4550))
* Paint Data: Send correct output after clicking Reset to Input ([#4551](../../pull/4551))
* TimeVariable.parse: Do not modify _ISO_FORMATS ([#4539](../../pull/4539))
* heatmap: Ensure minimim size for color annotations ([#4519](../../pull/4519))
* OWEditDomain: Clear editor when data is disconnected ([#4484](../../pull/4484))
* ContinuousPalettesModel: Disable 'category' items via `flags` ([#4538](../../pull/4538))
* utils/image: Return early when image is empty ([#4520](../../pull/4520))
* graphicstextlist: Use integer font metrics again ([#4524](../../pull/4524))
* Concatenate: Fix wrong merging of categorical features ([#4425](../../pull/4425))
* Ensure unique var names in file ([#4431](../../pull/4431))
* Rank: Fix error with Random forest ([#4457](../../pull/4457))
* owhierclustering: Update the scene's sceneRect ([#4459](../../pull/4459))
* Venn Diagram is slow for big datasets ([#4400](../../pull/4400))
* Fix missing values after purging unused values ([#4432](../../pull/4432))
* File: Construct unique column names. ([#4420](../../pull/4420))
* format_summary_details: Replace 'features' with 'variables' ([#4419](../../pull/4419))
* Feature Constructor: Catch exceptions ([#4401](../../pull/4401))
* Continuize: Disable normalizing sparse data ([#4379](../../pull/4379))
* Python script serialization state ([#4345](../../pull/4345))
* DataProjectionWidget: Update combos on new data ([#4405](../../pull/4405))
* DataProjectionWidget: attribute Selected ([#4393](../../pull/4393))
* Fix slow clear/delete in 'Heat Map' 'Hier. Clustering', 'Distance Map' ([#4365](../../pull/4365))
* Explicitly define the protocol version for pickling ([#4388](../../pull/4388))
* Table.from_table: fix caching with reused ids  ([#4370](../../pull/4370))
* FeatureStatistics: Convert selected rows to list ([#4375](../../pull/4375))
* Error message on tSNE with one variable ([#4364](../../pull/4364))
* Normalize: Integer variable representation ([#4350](../../pull/4350))
* DendrogramWidget: Remove event filters before removing items ([#4361](../../pull/4361))
* Round bhattacharayya ([#4340](../../pull/4340))


[3.24.1] - 2020-01-17
--------------------
##### Enhancements
* OWPreprocess: data info displayed in status bar ([#4333](../../pull/4333))
* OWDiscretize: data info displayed in status bar ([#4331](../../pull/4331))
* OWContinuize: data info displayed in status bar ([#4327](../../pull/4327))
* OWTranspose: data info displayed in status bar ([#4295](../../pull/4295))
* Silhouette plot: Accept distance matrix on input ([#4313](../../pull/4313))
* Preprocess: Add filtering by missing values ([#4266](../../pull/4266))
* Allow add-ons to register file format for the Save widget ([#4302](../../pull/4302))
* Box Plot: Add box for missing group values ([#4292](../../pull/4292))
* clustering/hierarchical: Use optimal\_leaf\_ordering from scipy ([#4288](../../pull/4288))
* Distributions: Add option to hide bars ([#4301](../../pull/4301))

##### Bugfixes
* ExcelReader: Speedup ([#4339](../../pull/4339))
* Nomogram: Adjust scale considering label width ([#4329](../../pull/4329))
* owhierarchicalclustering: Prescale dendrogram geometry ([#4322](../../pull/4322))
* table\_to\_frame: metas lost on conversion ([#4259](../../pull/4259))
* TestOWRank: Setting type should not be np.ndarray ([#4315](../../pull/4315))
* Distances: Fix restoring the cosine distance ([#4311](../../pull/4311))
* Fix saving workflows that contain a Rank widget ([#4289](../../pull/4289))
* utils/textimport: Remove 'exclusive' kwarg from QActionGroup call ([#4298](../../pull/4298))


[3.24.0] - 2019-12-20
--------------------
##### Enhancements
* Remove Variable.make ([#3925](../../pull/3925))
* OWTreeViewer: Bold predicted values in tree nodes ([#4269](../../pull/4269))
* Edit Domain: Reinterpret column type transforms ([#4262](../../pull/4262))
* OWTree: Add 'Classification Tree' keyword ([#4283](../../pull/4283))
* ConcurrentWidgetMixin: Cancel task on input change ([#4219](../../pull/4219))
* PCA: Add a signal with original data + pca ([#4255](../../pull/4255))
* Distances: Offload work to a separate thread ([#4046](../../pull/4046))
* Venn Diagram: Add relations over columns, simplify over rows ([#4006](../../pull/4006))
* Merge Data: Implement context settings ([#4248](../../pull/4248))
* Heatmap: Option to center color palette at 0 ([#4218](../../pull/4218))
* owheatmap: Add Split By combo box ([#4234](../../pull/4234))
* Heatmap: Allow labeling by any variable ([#4209](../../pull/4209))
* Test & Score Widget: Cancellability on input change. ([#4079](../../pull/4079))
* Boxplot no longer stretches bars when this is uninformative ([#4176](../../pull/4176))
* Box plot: Add 'order by importance' checkbox to groups ([#4055](../../pull/4055))
* Add pyproject.toml ([#4179](../../pull/4179))
* Add remove sparse features preprocessor ([#4093](../../pull/4093))
* Output to OWCorrespondence ([#4180](../../pull/4180))
* macOS: Installer python version ([#4130](../../pull/4130))
* OWBoxPlot: Show missing values ([#4135](../../pull/4135))
* Neighbors: Data info displayed in status bar ([#4157](../../pull/4157))
* Preprocess: Tests update
* Bhatthacharayya distance ([#4111](../../pull/4111))
* MergeData: Don't remove duplicate columns with different data ([#4100](../../pull/4100))
* Nice binning of time variables (Distributions, SOM) ([#4123](../../pull/4123))
* OWBaseSql: Base widget for connecting to a database ([#4083](../../pull/4083))
* k-Means: Add normalization checkbox ([#4099](../../pull/4099))
* Datasets: Remove control area ([#4071](../../pull/4071))
* Correlations: Include continuous class and meta variables ([#4067](../../pull/4067))

##### Bugfixes
* SQL: Save selected backend to settings ([#4270](../../pull/4270))
* ExcelReader: Migrate to openpyxl ([#4279](../../pull/4279))
* owcsvimport: Fix last/recent item serialization ([#4272](../../pull/4272))
* owselectcolumns: Fix move up/down type error ([#4271](../../pull/4271))
* Table: Keep pending selection if data is None ([#4281](../../pull/4281))
* Select Rows: Fix crash on changed variable type ([#4254](../../pull/4254))
* Louvain Clustering: Update graph output for compatibility with the new network add-on ([#4258](../../pull/4258))
* Merge data: Migrate settings ([#4263](../../pull/4263))
* Feature Statistics: Do not crash on empty domain ([#4245](../../pull/4245))
* File widget: fix name change ([#4235](../../pull/4235))
* Various fixes of box plot ([#4231](../../pull/4231))
* Fix guessing strategy for date and time variables ([#4226](../../pull/4226))
* owtable: Preserve tab order of updated inputs ([#4225](../../pull/4225))
* Feature Constructor: Compatibility with Python 3.8 ([#4222](../../pull/4222))
* File: Fix domain edit on no changes ([#4232](../../pull/4232))
* Normalizer: Retain attributes of attributes ([#4217](../../pull/4217))
* Hierarchical Clustering: Fix Annotations selection ([#4214](../../pull/4214))
* MDS: Place lines under points and labels ([#4213](../../pull/4213))
* Fix crash in Preprocess' Select Relevant Features when there are no features ([#4207](../../pull/4207))
* Louvain Clustering: fix setting to restore correctly ([#4187](../../pull/4187))
* SOM: Fix crash when clicking on empty canvas ([#4177](../../pull/4177))
* build-conda-installer.sh: Do not use python version from Miniconda ([#4142](../../pull/4142))
* Fix core dump in CI in SOM on sparse data ([#4174](../../pull/4174))
* ROC and LiftCurve: Store context settings ([#4138](../../pull/4138))
* Preprocess: Tests update
* Normalize: Fix crash with nan column ([#4125](../../pull/4125))
* Warning for discrete variable with >100 values in OWFile ([#4120](../../pull/4120))
* owcsvimport: Make the progress update a direct connection ([#4109](../../pull/4109))
* Scatterplot: Disable vizrank when features on input ([#4102](../../pull/4102))
* Sieve: Disable vizrank when features on input ([#4101](../../pull/4101))
* Datetime conversion to string ([#4098](../../pull/4098))
* Self-Organizing Map: Fix restoring width ([#4097](../../pull/4097))
* K-means: Save Silhouette Scores selection ([#4082](../../pull/4082))
* OWTestLearners: Vertically align to center ([#4095](../../pull/4095))
* Model's data_to_model_domain supports sparse matrix ([#4081](../../pull/4081))
* Color widget failed after removing Variable.make ([#4041](../../pull/4041))
* Deprecated processEvents argument in Test&Score ([#4047](../../pull/4047))
* Merge data: Rename variables with duplicated names ([#4076](../../pull/4076))
* k-Means: Impute missing values ([#4068](../../pull/4068))
* Predictions: Handle discrete target with no values ([#4066](../../pull/4066))
* Fix storing and retrieving selection, and unconditional auto commit ([#3957](../../pull/3957))
* Scatterplot: Enable/disable vizrank button ([#4016](../../pull/4016))
* Correlations: Add progress bar, retain responsiveness ([#4011](../../pull/4011))


[3.23.1] - 2019-10-03
--------------------
##### Bugfixes
* Addons install with conda by default


[3.23.0] - 2019-09-05
--------------------
##### Enhancements
* Pull YAML feed of notifications on startup, refactor notifications ([#3933](../../pull/3933))
* Widget for Self-Organizing Maps ([#3928](../../pull/3928))
* DBSCAN widget ([#3917](../../pull/3917))
* BoxPlot: Write the anova/t-test statistic onto the plot. ([#3945](../../pull/3945))
* Feature Constructor: Easier categorical features, allow creation of date/time, easier use of string data ([#3936](../../pull/3936))
* Merge data allows matching by multiple pairs of columns ([#3919](../../pull/3919))
* Sticky graphics header/footer views ([#3930](../../pull/3930))
* Shiny renewed widget Distributions ([#3896](../../pull/3896))
* Calibration plot (add performance curves) and a new Calibrated Learner widget ([#3881](../../pull/3881))
* Added Specificity as a new score in Test&Score ([#3907](../../pull/3907))
* Separate canvas and base widget ([#3772](../../pull/3772))

##### Bugfixes
* Conda Installer: Restore compatibility with latest anaconda python ([#4004](../../pull/4004))
* Scatter plot: Hidden variables fix ([#3985](../../pull/3985))
* Boxplot fix ([#3983](../../pull/3983))
* Heat map: Cannot cluster a single instance ([#3980](../../pull/3980))
* Test and Score: Sort numerically, not alphabetically ([#3951](../../pull/3951))
* OWProjectionWidgetBase: Update when domain is changed ([#3946](../../pull/3946))
* Change normalization to Scaling in SVM ([#3898](../../pull/3898))
* Restore usage tracking ([#3921](../../pull/3921))
* main: Fix widget settings directory path to clear ([#3932](../../pull/3932))
* Backcompatibility stubs ([#3926](../../pull/3926))
* OWNeighbours fix manual apply for some options ([#3911](../../pull/3911))
* Documentation links ([#3897](../../pull/3897))
* Update output on new input, even if auto commit is disabled ([#3844](../../pull/3844))


[3.22.0] - 2019-06-26
--------------------
##### Enhancements
* Unified clustering API ([#3814](../../pull/3814))
* t-SNE: Load openTSNE lazily" ([#3894](../../pull/3894))
* Replace popups with non-intrusive notifications ([#3855](../../pull/3855))
* CSV File Import widget ([#3876](../../pull/3876))
* t-SNE: Load openTSNE lazily ([#3883](../../pull/3883))
* Faster drawing in scatterplot ([#3871](../../pull/3871))
* Mosaic: Wrap legend ([#3866](../../pull/3866))
* Add conditions for all variables, or all numeric or textual variables ([#3836](../../pull/3836))
* Shared namespaces for PythonScript widgets ([#3840](../../pull/3840))
* Pivot: New widget ([#3823](../../pull/3823))
* Reset settings button ([#3795](../../pull/3795))
* WebviewWidget: expose JavaScript timeout limit ([#3811](../../pull/3811))

##### Bugfixes
* OWLinearProjection: limit axis ([#3885](../../pull/3885))
* Development readme ([#3889](../../pull/3889))
* OWRadviz: limit number of vars in RadvizVizRank ([#3886](../../pull/3886))
* OWCreateClass: Reuse variables created with same settings ([#3868](../../pull/3868))
* DistMatrix should return numpy datatypes ([#3865](../../pull/3865))
* OWRadviz: legible axis labels ([#3809](../../pull/3809))
* util: Fix bincount for object arrays ([#3831](../../pull/3831))
* condainstall: Fix env for running conda ([#3843](../../pull/3843))
* DBSCAN: Fix predicted labels ([#3833](../../pull/3833))
* Rare quickmenu crash fix ([#3832](../../pull/3832))
* OWWidget: Fix an error on mouse press when widget has no basic layout ([#3821](../../pull/3821))
* VerticalItemDelegate: Do not cut long vertical labels ([#3803](../../pull/3803))
* Minor improvements to pythagorean trees ([#3777](../../pull/3777))
* owmds: Fix test for error display/activate ([#3813](../../pull/3813))


[3.21.0] - 2019-05-20
---------------------
##### Enhancements
* Error Animations ([#3788](../../pull/3788))
* OWTSNE: Offload computation to separate thread ([#3604](../../pull/3604))
* OWDistributions: add cumulative distribution ([#3766](../../pull/3766))
* Edit Domain: Merge categorical values ([#3725](../../pull/3725))
* Transform: Values of primitive variables as feature names ([#3721](../../pull/3721))
* k-Means: Output centroid labels ([#3695](../../pull/3695))
* Support sparse Jaccard ([#3657](../../pull/3657))
* Offload work to a separate thread ([#3627](../../pull/3627))
* Correlations: Enhancements and fixes ([#3660](../../pull/3660))
* DomainEditor: Indicate changed variables with bold face font ([#3576](../../pull/3576))
* OWPythonScript: dropping and pasting of python scripts ([#3611](../../pull/3611))
* Improve Save widget's gui ([#3545](../../pull/3545))
* Table.from_numpy: Replace infs with nans. ([#3624](../../pull/3624))
* FDR: Calculate FDR using numpy ([#3625](../../pull/3625))
* Correlations: fixes and enhancements ([#3591](../../pull/3591))
* Preprocess: implement Select Relevant Feature's percentile ([#3588](../../pull/3588))
* Added keyboard shortcuts for Align & Freeze/Unfreeze  ([#3601](../../pull/3601))
* PCA: Remove SVD & add normalization for sparse ([#3581](../../pull/3581))
* OwLouvain: Add normalize data checkbox to PCA preprocessing ([#3573](../../pull/3573))
* Use %g (including sci notation) if number of decimals is not set ([#3574](../../pull/3574))
* PCA: Preserve f32s & reduce memory footprint when computing means ([#3582](../../pull/3582))

##### Bugfixes
* OWLinePlot: legible bottom axis labels ([#3768](../../pull/3768))
* OWPythagorasTree: Enable node selection from forests with categorical variables. ([#3775](../../pull/3775))
* canvas/help: Fix a NameError in exception handler ([#3759](../../pull/3759))
* ProjectionWidgetTestMixin: Fix test_plot_once ([#3738](../../pull/3738))
* LinearProjectionVizRank: Add a necessary check ([#3732](../../pull/3732))
* stats: Fix statistics for primitive variables ([#3722](../../pull/3722))
* SQL Table: Restore selected table from settings ([#3703](../../pull/3703))
* OWAnchorProjectionWidget: Retain valid_data when reloading dataset ([#3718](../../pull/3718))
* Compatibility with Logitech's Smart Move ([#3702](../../pull/3702))
* OWTable: Don't set selection when there is no data on input ([#3693](../../pull/3693))
* Save Data: Remove extra file extensions ([#3700](../../pull/3700))
* owheatmap: Fix group label size policy ([#3688](../../pull/3688))
* Test & Score: hide warnings for hidden scores. ([#3676](../../pull/3676))
* VizRankDialog: Use extended thread pool to prevent segfaults ([#3669](../../pull/3669))
* Send usage statistics in a thread at startup ([#3632](../../pull/3632))
* Transform: Replace 'Preprocess' input with 'Template Data' input ([#3673](../../pull/3673))
* OWTable: Include attributes from meta attributes in header row labels ([#3633](../../pull/3633))
* OWSieve: Fix operators in tooltips ([#3602](../../pull/3602))
* MDS: Handle subset data ([#3620](../../pull/3620))
* migrate_context: widgets crash when migrating context without version ([#3603](../../pull/3603))
* Louvain clustering fails when Table or ndarray on input. ([#3618](../../pull/3618))
* Orange: add more informatino on missing compiled libraries ([#3614](../../pull/3614))
* LinearProjection: Disable LDA for less than three classes ([#3615](../../pull/3615))
* Orange restart dialogs: Improve wording ([#3605](../../pull/3605))
* OWDistances: Use only binary features for Jaccard distance ([#3569](../../pull/3569))
* ScatterplotGraph: Use mapToView instead of mapRectFromParent ([#3571](../../pull/3571))
* MDS: Fix crashes when feeding column distances ([#3583](../../pull/3583))
* Naive Bayes: Ignore existing classes in Laplacian smoothing ([#3575](../../pull/3575))


[3.20.1] - 2019-02-12
--------------------
##### Enhancements
* t-SNE: Add Normalize data checkbox ([#3570](../../pull/3570))
* Louvain show number of clusters ([#3572](../../pull/3572))

##### Bugfixes
* t-SNE speed-ups ([#3592](../../pull/3592))
* setup.py: Specify python-louvain version constraint ([#3587](../../pull/3587))


[3.20.0] - 2019-02-01
--------------------
##### Enhancements
* Naive Bayes: Implement predict, fix predict_storage ([#3540](../../pull/3540))
* OWTransform: add option to keep original data #3526 ([#3549](../../pull/3549))
* Implement better randomized PCA ([#3532](../../pull/3532))
* Scatterplot: Draw separate regression lines for colors; add orthonormal regression ([#3518](../../pull/3518))
* Edit Domain: Add support for ordered categorical variables ([#3535](../../pull/3535))
* Warn when data subset is unrelated to data ([#3507](../../pull/3507))
* Label subset ([#3506](../../pull/3506))
* OWLouvain: Ensure deterministic clustering ([#3492](../../pull/3492))
* t-SNE: Updates 2. ([#3475](../../pull/3475))
* Save Data: Support saving to Excel ([#3453](../../pull/3453))
* OWLinePlot: Move from prototypes to core ([#3440](../../pull/3440))

##### Bugfixes
* Warning about a window title without a placeholder ([#3554](../../pull/3554))
* PaintData: Fix control area width ([#3560](../../pull/3560))
* OWWidget: Remove wheelEvent reimplementation ([#3557](../../pull/3557))
* Data sets: Remove class from 'bupa' ([#3556](../../pull/3556))
* Neighbours: Show error when data and reference have different domain. ([#3547](../../pull/3547))
* Feature statistics fixes ([#3480](../../pull/3480))
* Removed header types and flags from .csv and .tab ([#3427](../../pull/3427))
* Python script widget: prevent data loss  ([#3529](../../pull/3529))
* setup.cfg: Change the default for with-htmlhelp config option ([#3536](../../pull/3536))
* Don't add miniconda to path and register as system python ([#3525](../../pull/3525))
* OWDataProjectionWidget: check validity, fix sparse data reloading ([#3485](../../pull/3485))
* OWMergeData removes table meta attributes ([#3474](../../pull/3474))
* [MNT] Remove segfault in tests for building documentation ([#3491](../../pull/3491))
* OWFeatureStatistics: Fix scipy.stats.mode crash on sparse data ([#3488](../../pull/3488))
* OWDataProjectionWidget: Fix applying selection ([#3466](../../pull/3466))


[3.19.0] - 2018-12-11
--------------------
##### Enhancements
* Remove discrete attributes from scatter plot's axes ([#3434](../../pull/3434))
* OWScatterPlotBase: Animate dot resize ([#3436](../../pull/3436))
* Introduce stacking ([#3291](../../pull/3291))
* OWWidget: Input/output summary  ([#2556](../../pull/2556))
* File: Provide percent missing values in Info box ([#3305](../../pull/3305))
* OWHierarchicalClustering: Use selection indices for selection restore ([#3282](../../pull/3282))
* Data Info display data set name ([#3187](../../pull/3187))
* tSNE: Output preprocessor ([#3407](../../pull/3407))
* Pythagorean Tree: children order ([#3393](../../pull/3393))

##### Bugfixes
* RemoveNaNColumns: Move to preprocess ([#3464](../../pull/3464))
* Scatterplot Vizrank: Don't use discrete variables ([#3463](../../pull/3463))
* canvas/widgetsscheme: Remove use of QObject.destroyed for logging ([#3447](../../pull/3447))
* OWFeatureStatistics: Don't attempt to sort when no data on input ([#3449](../../pull/3449))
* Rank: Fix crash on dataset with missing values ([#3458](../../pull/3458))
* Radviz: Enable projection for less than two selected variables ([#3444](../../pull/3444))
* t-SNE: Generate temporary projection data ([#3454](../../pull/3454))
* Scatter Plot: Always setup plot ([#3450](../../pull/3450))
* Mosaic: Always reset discrete_data ([#3445](../../pull/3445))
* Save Data: Reset writer upon changing the extension ([#3437](../../pull/3437))
* Scatter Plot: Replot when input Features ([#3439](../../pull/3439))
* build-conda-installer: Update the included Miniconda installer ([#3429](../../pull/3429))
* OWDataProjectionWidget: Consider tables with nan-s equal ([#3435](../../pull/3435))
* Projections: Retain embedding if non-relevant variables change ([#3428](../../pull/3428))
* PCA: Rename components to PC1, PC2, PC3, ... ([#3423](../../pull/3423))


[3.18.0] - 2018-11-13
--------------------
##### Enhancements
* tSNE: Move from single-cell to core ([#3379](../../pull/3379))
* Transform: Add new widget ([#3346](../../pull/3346))
* Correlations: Move from prototypes to core ([#3362](../../pull/3362))
* Install widget help files ([#3345](../../pull/3345))
* Feature Statistics: Move from prototypes to core ([#3303](../../pull/3303))
* Replace scikit-learn tSNE with faster implementation ([#3192](../../pull/3192))
* Select Columns: Enable filtering of used features ([#3363](../../pull/3363))

##### Bugfixes
* setup.py: Remove trailing slash from directory names in data_files ([#3394](../../pull/3394))
* condainstall.bat: Add conda.bat and activate.bat scripts again ([#3389](../../pull/3389))
* LearnerScorer: fix for preprocessed data ([#3381](../../pull/3381))
* Feature Statistics: Update outputs on new data ([#3382](../../pull/3382))
* python-framework.sh: Fix 'Current' symlink creation ([#3373](../../pull/3373))
* Fix wrong indices in tooltips in projection when some data was invalid ([#3357](../../pull/3357))
* Scatterplot's VizRank no longer crashes in presence of nonprimitive metas ([#3347](../../pull/3347))
* Predictions: Fix failure after failed predictor ([#3337](../../pull/3337))
* Louvain Clustering: Do not invalidate output on PCA slider change with apply disabled ([#3339](../../pull/3339))
* Use minimal keyring implementation for tests ([#3359](../../pull/3359))
* OWFreeViz: Fix optimization for data with missing values ([#3358](../../pull/3358))


[3.17.0] - 2018-10-26
--------------------
##### Enhancements
* OWSelectAttributes: Use input features ([#3299](../../pull/3299))

##### Bugfixes
* OWDataTable: reset selections on domain change ([#3327](../../pull/3327))
* owlouvainclustering: Fix race conditions ([#3322](../../pull/3322))
* Save data widget crash on no data ([#3311](../../pull/3311))
* OWWidget: Preserve widget geometry between hide/show events ([#3304](../../pull/3304))
* Fix OWWidget destruction ([#3296](../../pull/3296))
* OWWidget: Fix size hint propagation ([#3253](../../pull/3253))


[3.16.0] - 2018-09-14
--------------------
##### Enhancements
* ROC analysis: show thresholds ([#3172](../../pull/3172))
* Edit Domain: Record transformations ([#3231](../../pull/3231))
* Data Table: Enable deselection ([#3189](../../pull/3189))
* Empty helper pane message ([#3210](../../pull/3210))
* Matplotlib output for Scatter plot ([#3134](../../pull/3134))
* Scatterplot: indicate overlap of points. ([#3177](../../pull/3177))
* Selection of format and compression in save data widget ([#3147](../../pull/3147))
* OWBoxPlot: add option to sort discrete distributions by size ([#3156](../../pull/3156))
* Table: speed-up computation of basic stats of given columns. ([#3166](../../pull/3166))
* Canvas: 'Window Groups' continued ([#3085](../../pull/3085))
* Combo Box Search Filter ([#3014](../../pull/3014))
* Widget Insertion ([#3179](../../pull/3179))

##### Bugfixes
* Documentation fetching with redirects ([#3248](../../pull/3248))
* DiscreteVariable reconstruction ([#3242](../../pull/3242))
* io: Handle mismatched number of header/data values ([#3237](../../pull/3237))
* OWNeuralNetwork model pickling ([#3230](../../pull/3230))
* Variable: Prevent hashing of Values of DiscreteVariable. ([#3217](../../pull/3217))


[3.15.0] - 2018-08-06
--------------------
##### Enhancements
* Silhouette Plot: Add cosine distance ([#3176](../../pull/3176))
* Add pandas_compat.table_to_frame(tab) ([#3180](../../pull/3180))
* OWSelectByDataIndex: New widget (move from Prototypes) ([#3181](../../pull/3181))
* Make filters available in Orange.data namespace. ([#3170](../../pull/3170))
* Move Louvain clustering from prototypes to core ([#3111](../../pull/3111))
* OWWidget: Collapse/expand the widget on control area toggle ([#3146](../../pull/3146))
* Rank: SklScorer should use the faster SklImputer. ([#3164](../../pull/3164))
* RecentFiles: Check for missing file in workflow dir ([#3064](../../pull/3064))
* Smart widget suggestions ([#3112](../../pull/3112))
* Match Keywords in Widget Search ([#3117](../../pull/3117))
* io: Speedup write_data ([#3115](../../pull/3115))
* OWEditDomain: Enable reordering of discrete variables ([#3119](../../pull/3119))

##### Bugfixes
* oweditdomain: Fix an IndexError when all rows are deselected ([#3183](../../pull/3183))
* OWFreeViz: fix class density size ([#3158](../../pull/3158))
* OWBoxPlot: Fix empty continuous contingency check ([#3165](../../pull/3165))
* OWSql: enforce data download for non PostgreSQL databases ([#3178](../../pull/3178))
* owlouvainclustering: Make the task completion handler a single slot ([#3182](../../pull/3182))
* OWReport: disable save and print on empty report ([#3175](../../pull/3175))
* RemoveConstant: remove constant NaN features. ([#3163](../../pull/3163))
* utils/concurrent: Switch default thread pool ([#3138](../../pull/3138))
* OWBoxPlot: Fix quartiles computation ([#3159](../../pull/3159))
* OWBoxPlot: Plot axis labels and quartiles label layout ([#3162](../../pull/3162))
* [RFC] OWHeatMap: remove empty clusters from visualization ([#3155](../../pull/3155))
* report: Fix the number of hidden rows. ([#3150](../../pull/3150))
* [RFC] KMeans upgrade sparse support ([#3140](../../pull/3140))
* WebView fixes ([#3148](../../pull/3148))
* ci/appveyor: Update test dependencies ([#3139](../../pull/3139))
* Replace use of obsolete QStyle.standardPixmap ([#3127](../../pull/3127))
* BoxPlot: Hide groups with no instances ([#3122](../../pull/3122))


[3.14.0] - 2018-07-04
--------------------
##### Enhancements
* MergeData: add autocommit button ([#3091](../../pull/3091))
* Canvas: Window Groups ([#3066](../../pull/3066))
* Save data with compression ([#3047](../../pull/3047))
* Neural network widget that works in a separate thread ([#2958](../../pull/2958))
* Display Widgets on Top option ([#3038](../../pull/3038))
* Implement multi window editing ([#2820](../../pull/2820))
* Data Info widget displays data attributes ([#3022](../../pull/3022))
* Icon redesign: k-means, clustering, distances ([#3018](../../pull/3018))

##### Bugfixes
* postgres: Fix wrong discrete values ([#3109](../../pull/3109))
* OWRank: Select Attributes fixes and improvements ([#3084](../../pull/3084))
* EditDomain: Editing TimeVariable broke formatting ([#3101](../../pull/3101))
* OWMosaic: Don't offer String meta attributes ([#3072](../../pull/3072))
* owkmeans: fix initialization choice ([#3090](../../pull/3090))
* Workaround for segfaults with Nvidia on Linux ([#3100](../../pull/3100))
* Canvas: Fix 'Widgest on top' ([#3068](../../pull/3068))
* Re-cythonize with Cython 0.28 for Python 3.7 compatibility ([#3067](../../pull/3067))
* BSD compatibility patch ([#3061](../../pull/3061))
* OWScatterOWScatterPlotGraph: Match group colors with marker colors ([#3053](../../pull/3053))
* listfilter: Fix filter line edit completion ([#2896](../../pull/2896))
* VizRank: Fix race condition in `toggle` ([#3042](../../pull/3042))
* Heat Map: Allow labeling by TimeVariable ([#3026](../../pull/3026))
* Select Columns: Drag/drop ([#3032](../../pull/3032))
* gui: Suppress mouse button release on the combo box popup ([#3025](../../pull/3025))
* tests: Fix time tracking in process_events ([#3041](../../pull/3041))
* test_owmosaic: Cleanup/join threads on test tear down ([#3040](../../pull/3040))
* owselectcolumns: Fix performance on filtering with selection ([#3030](../../pull/3030))
* test: Fix tests for 'Datasets' widget ([#3033](../../pull/3033))
* Sorting add-ons in the alphabetical order ([#3013](../../pull/3013))
* owscatterplot: Use correct data for regression line ([#3024](../../pull/3024))
* Add-ons dialog: Restore state ([#3017](../../pull/3017))
* Feature constructor does not restore features when loading from saved workflow ([#2996](../../pull/2996))
* boxplot labels overlap ([#3011](../../pull/3011))
* owdiscretize: Fix quadratic complexitiy in the n variables ([#3016](../../pull/3016))


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


[next]: https://github.com/biolab/orange3/compare/3.31.0...HEAD
[3.31.0]: https://github.com/biolab/orange3/compare/3.30.2...3.31.0
[3.30.2]: https://github.com/biolab/orange3/compare/3.30.1...3.30.2
[3.30.1]: https://github.com/biolab/orange3/compare/3.30.0...3.30.1
[3.30.0]: https://github.com/biolab/orange3/compare/3.29.3...3.30.0
[3.29.3]: https://github.com/biolab/orange3/compare/3.29.2...3.29.3
[3.29.2]: https://github.com/biolab/orange3/compare/3.29.1...3.29.2
[3.29.1]: https://github.com/biolab/orange3/compare/3.29.0...3.29.1
[3.29.0]: https://github.com/biolab/orange3/compare/3.28.0...3.29.0
[3.28.0]: https://github.com/biolab/orange3/compare/3.27.1...3.28.0
[3.27.1]: https://github.com/biolab/orange3/compare/3.27.0...3.27.1
[3.27.0]: https://github.com/biolab/orange3/compare/3.26.0...3.27.0
[3.26.0]: https://github.com/biolab/orange3/compare/3.25.1...3.26.0
[3.25.1]: https://github.com/biolab/orange3/compare/3.25.0...3.25.1
[3.25.0]: https://github.com/biolab/orange3/compare/3.24.1...3.25.0
[3.24.1]: https://github.com/biolab/orange3/compare/3.24.0...3.24.1
[3.24.0]: https://github.com/biolab/orange3/compare/3.23.1...3.24.0
[3.23.1]: https://github.com/biolab/orange3/compare/3.23.0...3.23.1
[3.23.0]: https://github.com/biolab/orange3/compare/3.22.0...3.23.0
[3.22.0]: https://github.com/biolab/orange3/compare/3.21.0...3.22.0
[3.21.0]: https://github.com/biolab/orange3/compare/3.20.1...3.21.0
[3.20.1]: https://github.com/biolab/orange3/compare/3.20.0...3.20.1
[3.20.0]: https://github.com/biolab/orange3/compare/3.19.0...3.20.0
[3.19.0]: https://github.com/biolab/orange3/compare/3.18.0...3.19.0
[3.18.0]: https://github.com/biolab/orange3/compare/3.17.0...3.18.0
[3.17.0]: https://github.com/biolab/orange3/compare/3.16.0...3.17.0
[3.16.0]: https://github.com/biolab/orange3/compare/3.15.0...3.16.0
[3.15.0]: https://github.com/biolab/orange3/compare/3.14.0...3.15.0
[3.14.0]: https://github.com/biolab/orange3/compare/3.13.0...3.14.0
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
