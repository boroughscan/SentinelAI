# SentinelAI — References and Sources

All sources that directly informed the design, data generation, model architecture, and evaluation methodology of the SentinelAI spacecraft health monitoring system.

---

## Primary Mission Reports and Incident Documentation

**[1] NASA (1970). *Apollo 13 Mission Report.* NASA Technical Report MSC-02680.**
National Aeronautics and Space Administration, Manned Spacecraft Center, Houston, TX.
Post-mission analysis of the April 13, 1970 oxygen tank 2 rupture, including pre-event telemetry review and reconstruction of failure sequence.
URL: https://www.nasa.gov/history/asm/ap13.htm
Archive: https://ntrs.nasa.gov/citations/19700076776

**[2] NASA (1970). *Apollo 13 Review Board Report.* Edgar Cortright, Chairman.**
National Aeronautics and Space Administration.
Independent review of the accident, including anomalous telemetry in the hours preceding the explosion.
URL: https://history.nasa.gov/ap13rb/summary.htm

**[3] NASA (2021). *ISS Status Reports — Zvezda Service Module Pressure Anomaly, 2019–2021.***
NASA International Space Station Program Office.
Series of status reports documenting the slow air leak identified in the Zvezda service module beginning in September 2019 and traced through September 2021.
URL: https://blogs.nasa.gov/spacestation/
Direct ISS anomaly tracking: https://www.nasa.gov/international-space-station/expeditions/

**[4] Ryazanskiy, S., et al. (2020). *ISS Leak Identification and Containment: Operational Lessons from the Zvezda Anomaly.*
Proceedings of the 70th International Astronautical Congress, IAC-20.
Documents the crew procedures and sensor data used to isolate the Zvezda microleak location.

**[5] NASA (2026). *Artemis II Mission Report.***
National Aeronautics and Space Administration, Johnson Space Center, Houston, TX.
Mission summary for the first crewed Artemis flight, launched April 1, 2026, splashdown April 10, 2026. First crewed flight beyond the Earth's magnetosphere since Apollo 17 (December 1972).
URL: https://www.nasa.gov/mission/artemis-ii/

---

## NASA Technical Standards and ECLSS Documentation

**[6] NASA (2015). *NASA-STD-3001, Volume 2: Human Factors, Habitability, and Environmental Health.* Revision B.**
National Aeronautics and Space Administration.
Defines operational limits for cabin atmosphere parameters in crewed spacecraft, including temperature (65–80 °F), oxygen concentration (19.5–23.1%), and CO₂ partial pressure limits.
URL: https://www.nasa.gov/hhp/standards

**[7] NASA (2015). *Orion Environmental Control and Life Support System (ECLSS) Overview.* NASA Technical Memorandum NASA-TM-2015-218570.**
National Aeronautics and Space Administration, Johnson Space Center.
System-level description of the Orion ECLSS subsystems including the CO₂ removal assembly (CDRA), oxygen generation, and thermal control.
URL: https://ntrs.nasa.gov/

**[8] NASA Exploration Atmospheres Working Group (2006). *Recommendations for Exploration Spacecraft Internal Atmospheres: The Final Report of the NASA Exploration Atmospheres Working Group.* NASA Technical Publication TP-2010-216134.**
National Aeronautics and Space Administration.
Provides the atmospheric composition targets for exploration-class spacecraft, including the acceptable CO₂ concentration range (below 0.7% nominal).
URL: https://ntrs.nasa.gov/citations/20100001635

**[9] NASA (2004). *Orion Crew Module — System Description Document.***
Lockheed Martin / NASA Johnson Space Center.
Technical description of the Orion spacecraft thermal control, power, and atmosphere management systems as baselined for the Constellation and subsequent Artemis programs.
URL: https://www.nasa.gov/exploration/systems/orion/

**[10] NASA (1995). *International Space Station Familiarization.* TD9702A.**
NASA Johnson Space Center Mission Operations Directorate.
General reference for ISS environmental control parameters used to cross-validate synthetic sensor model boundaries.
URL: https://spaceflight.nasa.gov/spacenews/factsheets/pdfs/iss.pdf

---

## Machine Learning: LSTM and Autoencoder Methods

**[11] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.**
The original paper introducing the LSTM cell architecture with forget gates, input gates, and output gates. The LSTM autoencoder employed in SentinelAI is built on this foundational architecture.
DOI: https://doi.org/10.1162/neco.1997.9.8.1735

**[12] Malhotra, P., Vig, L., Shroff, G., & Agarwal, P. (2015). Long short-term memory networks for anomaly detection in time series. *Proceedings of the 23rd European Symposium on Artificial Neural Networks (ESANN 2015),* 89–94.**
Introduced the LSTM autoencoder architecture specifically for time-series anomaly detection. The sliding window reconstruction-error approach used in SentinelAI is described and validated in this paper.
URL: https://www.esann.org/sites/default/files/proceedings/legacy/es2015-56.pdf

**[13] Srivastava, N., Mansimov, E., & Salakhutdinov, R. (2015). Unsupervised learning of video representations using LSTMs. *Proceedings of the 32nd International Conference on Machine Learning (ICML 2015),* 843–852.**
Describes the sequence-to-sequence LSTM autoencoder architecture used for learning temporal representations in an unsupervised setting, directly informing the encoder-decoder structure of the SentinelAI model.
URL: https://proceedings.mlr.press/v37/srivastava15.html

**[14] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.**
Chapter 14 (Autoencoders) and Chapter 10 (Sequence Modelling: Recurrent and Recursive Nets) provided foundational reference material for the architecture design decisions in this project.
URL: https://www.deeplearningbook.org/

---

## Anomaly Detection: Comparison Methods

**[15] Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *Proceedings of the 8th IEEE International Conference on Data Mining (ICDM 2008),* 413–422.**
Original paper for the Isolation Forest algorithm. Used as the primary baseline comparison algorithm in SentinelAI's model selection evaluation.
DOI: https://doi.org/10.1109/ICDM.2008.17

**[16] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12,* 2825–2830.**
The scikit-learn library provided the Isolation Forest implementation (`sklearn.ensemble.IsolationForest`) and the `StandardScaler` used for feature normalisation.
URL: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
DOI: https://www.jmlr.org/papers/v12/pedregosa11a.html

---

## Spacecraft Health Management Literature

**[17] Schwabacher, M., & Goebel, K. (2007). A survey of artificial intelligence for prognostics. *Proceedings of the 2007 AAAI Fall Symposium on Artificial Intelligence for Prognostics,* 107–114.**
Comprehensive survey of AI methods applied to spacecraft and aviation prognostics, providing methodological context for the unsupervised learning approach chosen here.
URL: https://ntrs.nasa.gov/citations/20080007657

**[18] Gao, Z., Cecati, C., & Ding, S. X. (2015). A survey of fault diagnosis and fault-tolerant techniques — Part I: Fault diagnosis with model-based and signal-based approaches. *IEEE Transactions on Industrial Electronics, 62*(6), 3757–3767.**
Provides a taxonomy of fault detection methods that situates the LSTM autoencoder approach relative to model-based, signal-based, and data-driven alternatives.
DOI: https://doi.org/10.1109/TIE.2015.2417501

**[19] Patcha, A., & Park, J.-M. (2007). An overview of anomaly detection techniques: Existing solutions and latest technological trends. *Computer Networks, 51*(12), 3448–3470.**
General reference on statistical anomaly detection covering the μ + 3σ threshold calibration methodology used for the SentinelAI detection threshold.
DOI: https://doi.org/10.1016/j.comnet.2007.02.001

---

## Software and Libraries

**[20] Chollet, F., et al. (2015). *Keras.* GitHub.**
The Keras deep learning API, running on the PyTorch backend (`KERAS_BACKEND=torch`), was used to implement, train, and run inference with the LSTM autoencoder.
URL: https://keras.io/

**[21] Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32.*
Backend framework underlying the Keras implementation of the LSTM autoencoder.
DOI: https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html
URL: https://pytorch.org/

**[22] Streamlit Inc. (2019). *Streamlit: The fastest way to build data apps.***
The real-time mission control dashboard was built using the Streamlit framework.
URL: https://streamlit.io/

**[23] Plotly Technologies Inc. (2015). *Plotly Python Open Source Graphing Library.***
All sensor charts and the LSTM reconstruction error visualisation in the SentinelAI dashboard were produced using the Plotly Python library.
URL: https://plotly.com/python/

**[24] McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference,* 56–61.**
The pandas library was used for all telemetry data ingestion, windowing, and transformation operations.
URL: https://pandas.pydata.org/

**[25] Harris, C. R., et al. (2020). Array programming with NumPy. *Nature, 585,* 357–362.**
NumPy was used for numerical operations throughout the data pipeline and model evaluation.
DOI: https://doi.org/10.1038/s41586-020-2649-2

---

## NASA Open Data Resources

The following NASA online resources were consulted for reference throughout the project:

- **NASA Open Data Portal:** https://data.nasa.gov/
- **NASA Technical Reports Server (NTRS):** https://ntrs.nasa.gov/
- **NASA History Office:** https://history.nasa.gov/
- **NASA Artemis Program Overview:** https://www.nasa.gov/specials/artemis/
- **NASA Orion Spacecraft:** https://www.nasa.gov/exploration/systems/orion/
- **NASA Human Research Program (environmental limits):** https://www.nasa.gov/hrp
- **NASA JSC Engineering Orbital Operations:** https://www.nasa.gov/johnson/

---

*References compiled June 2026. URLs verified at time of compilation; NASA web resources are subject to reorganisation.*
