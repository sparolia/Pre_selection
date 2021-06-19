import numpy
import math
import sys
import ROOT

ROOT.ROOT.EnableImplicitMT()

#Filename_tau = "/scratch/parolia/2021Mar23/BcToJPsiMuMu_is_jpsi_tau_merged.root"
#Filename_mu = "/scratch/parolia/2021Mar23/BcToJPsiMuMu_is_jpsi_mu_merged.root"

#Filename_sig = "/scratch/parolia/2021Mar23/BcToJPsiMuMu_is_jpsi_lepton_weight_v3.root"
Filename_sig = "/scratch/parolia/2021May27/BcToJPsiMuMu_is_jpsi_lepton_withWeights.root"
Filename_bkg = "/scratch/parolia/2021May27/mc_bkg_all_withWeights.root"
#Filename_bkg = "/scratch/parolia/2021Mar23/mc_bkg_all_withFlag.root"
#/scratch/parolia/2021May27
'''
Filename_bkg1 = "/scratch/parolia/2021Mar23/BcToJPsiMuMu_is_jpsi_pi_merged.root"
Filename_bkg2 = "/scratch/parolia/2021Mar23/BcToJPsiMuMu_is_jpsi_3pi_merged.root"
Filename_bkg3 = "/scratch/parolia/2021Mar23/BcToJPsiMuMu_is_jpsi_hc_merged.root"
Filename_bkg4 = "/scratch/parolia/2021Mar23/BcToJPsiMuMu_is_psi2s_mu_merged.root"
Filename_bkg5 = "/scratch/parolia/2021Mar23/BcToJPsiMuMu_is_psi2s_tau_merged.root"
Filename_bkg6 = "/scratch/parolia/2021Mar23/BcToJPsiMuMu_is_hc_mu_merged.root"
Filename_bkg7 = "/scratch/parolia/2021Mar23/BcToJPsiMuMu_is_chic0_mu_merged.root"
Filename_bkg8 = "/scratch/parolia/2021Mar23/BcToJPsiMuMu_is_chic1_mu_merged.root"
Filename_bkg9 = "/scratch/parolia/2021Mar23/BcToJPsiMuMu_is_chic2_mu_merged.root"
Filename_bkg10 = "/scratch/parolia/2021Mar23/HbToJPsiMuMu3MuFilter_ptmax_merged.root"
Filename_bkg11 = "/scratch/parolia/2021Mar23/HbToJPsiMuMu_ptmax_merged.root"
'''
tree_sig = ROOT.RDataFrame("BTo3Mu",Filename_sig)
tree_bkg = ROOT.RDataFrame("BTo3Mu",Filename_bkg)
#tree_bkg = ROOT.RDataFrame("BTo3Mu",{Filename_bkg1,Filename_bkg3,Filename_bkg4,Filename_bkg6,Filename_bkg7,Filename_bkg8,Filename_bkg9,Filename_bkg10,Filename_bkg11})


tree_sig_1 = tree_sig.Filter('k_mediumID > 0.5 && (mu1_mediumID > 0.5 && mu2_mediumID > 0.5)')
tree_sig_1 = tree_sig_1.Filter('mu1_isFromMuT &&( mu2_isFromMuT && k_isFromMuT)')
tree_sig_2 = tree_sig_1.Filter('Bmass < 6.2')

tree_bkg_1 = tree_bkg.Filter('k_mediumID > 0.5 && (mu1_mediumID>0.5 && mu2_mediumID>0.5)')
tree_bkg_1 = tree_bkg_1.Filter('mu1_isFromMuT &&( mu2_isFromMuT && k_isFromMuT)')
tree_bkg_2 = tree_bkg_1.Filter('Bmass < 6.2')


tree_sig_3 = tree_sig_2.Define("jpsimass_reso1","float x= 0.0; if(abs(mu1eta) < 1.2 && abs(mu2eta) < 1.2) x = (jpsi_mass - 3.0969); return(x)")\
                   .Filter("jpsimass_reso1 < 0.08")\
                   .Define("jpsimass_reso2","float x = 0.0; if(abs(mu1eta) > 1.2 && abs(mu2eta) > 1.2) x = (jpsi_mass - 3.0969); return (x)")\
                   .Filter("jpsimass_reso2 < 0.1")\
                   .Define("jpsimass_reso3","float x = 0.0; if((1.2 < abs(mu1eta) and abs(mu2eta) < 1.2) or (1.2 < abs(mu2eta) and abs(mu1eta) < 1.2)) x = (jpsi_mass - 3.0969); return (x)")\
                   .Filter("jpsimass_reso3 < 0.1")\

#tree_sig_1.Snapshot("BTo3Mu","Sig_basic.root")
tree_sig_3.Snapshot("BTo3Mu","Sig.root")

tree_bkg_3 = tree_bkg_2.Define("jpsimass_reso1","float x= 0.0; if(abs(mu1eta) < 1.2 && abs(mu2eta) < 1.2) x = (jpsi_mass - 3.0969); return(x)")\
                   .Filter("jpsimass_reso1 < 0.08")\
                   .Define("jpsimass_reso2","float x = 0.0; if(abs(mu1eta) > 1.2 && abs(mu2eta) > 1.2) x = (jpsi_mass - 3.0969); return (x)")\
                   .Filter("jpsimass_reso2 < 0.1")\
                   .Define("jpsimass_reso3","float x = 0.0; if((1.2 < abs(mu1eta) and abs(mu2eta) < 1.2) or (1.2 < abs(mu2eta) and abs(mu1eta) < 1.2)) x = (jpsi_mass - 3.0969); return (x)")\
                   .Filter("jpsimass_reso3 < 0.1")\

#tree_bkg_3 = tree_bkg_3.Define("is_signal_channel","int x = -1; return(x)")

#tree_bkg_1.Snapshot("BTo3Mu","Bkg_basic.root")
tree_bkg_3.Snapshot("BTo3Mu","Bkg.root")

tree_tau = tree_sig.Filter("is_signal_channel > 0.5") 
tree_mu = tree_sig.Filter("is_signal_channel < 0.5") 
tree_tau_1 = tree_sig_1.Filter("is_signal_channel > 0.5") 
tree_mu_1 = tree_sig_1.Filter("is_signal_channel < 0.5") 
tree_tau_2= tree_sig_2.Filter("is_signal_channel > 0.5") 
tree_mu_2 = tree_sig_2.Filter("is_signal_channel < 0.5") 
tree_tau_3 = tree_sig_3.Filter("is_signal_channel > 0.5") 
tree_mu_3 = tree_sig_3.Filter("is_signal_channel < 0.5") 

tree_tau_3.Snapshot("BTo3Mu","Sig_tau.root")
tree_mu_3.Snapshot("BTo3Mu","Sig_mu.root")

'''
#temp_1 = tree_sig_1.Define("ip3d_ns","float x=0.0; if((ip3d/ip3d_e)<0) x= -(ip3d/ip3d_e); else x = -10.0; return (x)")\
                #   .Define("ip3d_ps","float x=0.0; if((ip3d/ip3d_e)>0) x= (ip3d/ip3d_e); else x = -10.0; return (x)")
temp_1.Snapshot("BTo3Mu","Sig.root")

#temp_2 = tree_bkg_1.Define("ip3d_ns","float x=0.0; if((ip3d/ip3d_e)<0) x= -(ip3d/ip3d_e); else x = -10.0; return (x)")\
                 #  .Define("ip3d_ps","float x=0.0; if((ip3d/ip3d_e)>0) x= (ip3d/ip3d_e); else x = -10.0; return (x)")
temp_2.Snapshot("BTo3Mu","Bkg.root")
'''

entries1 = tree_sig.Count();
entries2 = tree_sig_1.Count();
entries3 = tree_sig_2.Count();
entries4 = tree_sig_3.Count();
entries5 = tree_bkg.Count();
entries6 = tree_bkg_1.Count();
entries7 = tree_bkg_2.Count();
entries8 = tree_bkg_3.Count();
entries9 = tree_tau.Count();
entries10 = tree_tau_1.Count();
entries11 = tree_tau_2.Count();
entries12 = tree_tau_3.Count();
entries13 = tree_mu.Count();
entries14 = tree_mu_1.Count();
entries15 = tree_mu_2.Count();
entries16 = tree_mu_3.Count();

print (entries1.GetValue())
print (entries2.GetValue())
print (entries3.GetValue())
print (entries4.GetValue())
print (entries5.GetValue())
print (entries6.GetValue())
print (entries7.GetValue())
print (entries8.GetValue())
print (entries9.GetValue())
print (entries10.GetValue())
print (entries11.GetValue())
print (entries12.GetValue())
print (entries13.GetValue())
print (entries14.GetValue())
print (entries15.GetValue())
print (entries16.GetValue())
