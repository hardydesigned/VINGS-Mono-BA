  Einzelner Run (Bonn):
  conda activate vings                                                                                                                                                                    
  cd /root/VINGS-Mono-BA
  PYTHONPATH=scripts python scripts/run.py configs/local/bonn_crowd.yaml                                                                                                                  
                  
  Mit Metriken (empfohlen):                                                                                                                                                               
  conda activate vings
  cd /root/VINGS-Mono-BA                                                                                                                                                                  
  ONLY="bonn_crowd" bash scripts/run_all_vings.sh