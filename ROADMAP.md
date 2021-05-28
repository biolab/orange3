# Orange Roadmap

In the next few years, we plan to expand Orange in the following directions:

- __Scalability__. Analyzing very large data in Orange is a challenge. Orange is an interactive data analytic tool with a flexible, user-defined workflow, which makes optimization for large data more complex than in the case of tools, which are either non-interactive or have a fixed data processing flow. We plan to address the following issues:
    - __Memory efficiency__. Orange can currently address data of up to around 500 MB.
    - __Speed-up of complex operations__. Hour-long processing is not acceptable for interactive data analysis.
    - __Interruptibility of processes and progress updates__. Operations that take too long are not always interruptible and may require shutting down the application; those same operations do not provide updates on progress reports.

- __Smoothness of the learning curve for new users__. This is not about fixing a problem but about building on Orange’s existing strength.
    - __Provide a tutorial for first-time users__ upon starting Orange. An excellent example is a guide in Open Street Map editor.
    - __Simplification of the interface__. Base Orange currently has under 200 components, and we keep this number low by design. Still, beginners should see fewer and be provided with assistance in further exploring.
    - __Gamification__. We are already exploring ways to implement self-paced tutorials within Orange and design ways to incorporate guidance for workflow construction, problem-driven exercises, and progress badges.
    - __Video content__. We will update and refresh existing [YouTube videos](http://youtube.com/orangedatamining) and record new ones.
    - __Lecture notes__. We have lecture notes for almost a hundred hands-on workshops carried out around the world. They were typeset in Pages, which worked nicely in the beginning, but now the management of about 300 pages of material became cumbersome. We are converting the lecture notes to LaTeX, enabling fast assembly of notes for a custom-designed hands-on workshop. The current solution needs improvements in design, and text needs updates and additional proofreading.

- __Improved functionality__ in several application fields, with particular priority being
    - __Spectroscopy__: we are making continuous progress with an add-on, thanks to support from [Synchrotron SOLEIL](https://www.synchrotron-soleil.fr/en).
    - __Text mining__: our goal is to design components for document characterization, summarization, and explanation of point-based visualizations. This work is in progress, thanks to the support by the [Slovene Ministry of Public Administration](https://www.gov.si/en/state-authorities/ministries/ministry-of-public-administration/).
    - __Network analysis__: speed-up and beautify existing visualizations and extend the toolbox with new network analysis procedures. This is being done in collaboration with the [International Laboratory for Applied Network Research, Higher Schools of Economics, Moscow](https://anr.hse.ru/en/).
    - __Time series analysis__: replace HighCharts with Qt-based visualizations, enable simultaneous analysis of a group of signals, improve modeling procedures, implement model evaluation techniques, and incorporate deep learning and auto-encoding for time series characterization, feature engineering, and prediction.

We need to address __funding__. The development of Orange relies on a core team sponsored by research and development grants. We will continue searching for new funding opportunities and seeking international collaborations to enrich Orange with new functionalities and improved interfaces. We seek funds both in developing Orange as a tool to democratize data science, assist in training machine learning, and support problem-solving in science and industry.
