import functions
import helper_tmva
import helpers
import ROOT
import argparse
import logging

import helper_jetclustering
import helper_flavourtagger

logger = logging.getLogger("fcclogger")

parser = functions.make_def_argparser()
args = parser.parse_args()
functions.set_threads(args)

functions.add_include_file("analyses/higgs_mass_xsec/functions.h")
functions.add_include_file("analyses/higgs_mass_xsec/functions_gen.h")


# define histograms
bins_score = (100, 0, 1)

bins_m = (250, 0, 250)
bins_p = (200, 0, 200)
bins_m_zoom = (200, 110, 130) # 100 MeV


bins_theta = (500, 0, 5)
bins_phi = (400, -4, 4)

bins_count = (100, 0, 100)
bins_pdgid = (60, -30, 30)
bins_charge = (10, -5, 5)

bins_resolution = (10000, 0.95, 1.05)
bins_resolution_1 = (20000, 0, 2)

jet_energy = (1000, 0, 100) # 100 MeV bins
dijet_m = (2000, 0, 200) # 100 MeV bins
visMass = (2000, 0, 200) # 100 MeV bins
missEnergy  = (2000, 0, 200) # 100 MeV bins

dijet_m_final = (500, 50, 100) # 100 MeV bins

bins_cos = (100, -1, 1)
bins_aco = (1000,0,1)
bins_cosThetaMiss = (10000, 0, 1)

bins_dR = (1000, 0, 10)

#jet clustering applications
njets = 4 # number of jets to be clustered
jetClusteringHelper4 = helper_jetclustering.ExclusiveJetClusteringHelper(njets, collection="ReconstructedParticles")
jetFlavourHelper = helper_flavourtagger.JetFlavourHelper(jetClusteringHelper4.jets, jetClusteringHelper4.constituents)
path = "data/flavourtagger/fccee_flavtagging_edm4hep_wc_v1"
jetFlavourHelper.load(f"{path}.json", f"{path}.onnx")
def build_graph(df, dataset):

    logging.info(f"build graph {dataset.name}")
    results, cols = [], []

    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")
    df = helpers.defineCutFlowVars(df) # make the cutX=X variables
    
    # define collections
    df = df.Alias("Particle0", "Particle#0.index")
    df = df.Alias("Particle1", "Particle#1.index")
    df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
    df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")


    # muons
    df = df.Alias("Muon0", "Muon#0.index")
    df = df.Define("muons_all", "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)")
    df = df.Define("muons_all_p", "FCCAnalyses::ReconstructedParticle::get_p(muons_all)")
    df = df.Define("muons_all_theta", "FCCAnalyses::ReconstructedParticle::get_theta(muons_all)")
    df = df.Define("muons_all_phi", "FCCAnalyses::ReconstructedParticle::get_phi(muons_all)")
    df = df.Define("muons_all_q", "FCCAnalyses::ReconstructedParticle::get_charge(muons_all)")
    df = df.Define("muons_all_no", "FCCAnalyses::ReconstructedParticle::get_n(muons_all)")

    df = df.Define("muons", "FCCAnalyses::ReconstructedParticle::sel_p(25)(muons_all)")
    df = df.Define("muons_p", "FCCAnalyses::ReconstructedParticle::get_p(muons)")
    df = df.Define("muons_theta", "FCCAnalyses::ReconstructedParticle::get_theta(muons)")
    df = df.Define("muons_phi", "FCCAnalyses::ReconstructedParticle::get_phi(muons)")
    df = df.Define("muons_q", "FCCAnalyses::ReconstructedParticle::get_charge(muons)")
    df = df.Define("muons_no", "FCCAnalyses::ReconstructedParticle::get_n(muons)")

    
    # electrons
    df = df.Alias("Electron0", "Electron#0.index")
    df = df.Define("electrons_all", "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)")
    df = df.Define("electrons_all_p", "FCCAnalyses::ReconstructedParticle::get_p(electrons_all)")
    df = df.Define("electrons_all_theta", "FCCAnalyses::ReconstructedParticle::get_theta(electrons_all)")
    df = df.Define("electrons_all_phi", "FCCAnalyses::ReconstructedParticle::get_phi(electrons_all)")
    df = df.Define("electrons_all_q", "FCCAnalyses::ReconstructedParticle::get_charge(electrons_all)")
    df = df.Define("electrons_all_no", "FCCAnalyses::ReconstructedParticle::get_n(electrons_all)")

    df = df.Define("electrons", "FCCAnalyses::ReconstructedParticle::sel_p(25)(electrons_all)")
    df = df.Define("electrons_p", "FCCAnalyses::ReconstructedParticle::get_p(electrons)")
    df = df.Define("electrons_theta", "FCCAnalyses::ReconstructedParticle::get_theta(electrons)")
    df = df.Define("electrons_phi", "FCCAnalyses::ReconstructedParticle::get_phi(electrons)")
    df = df.Define("electrons_q", "FCCAnalyses::ReconstructedParticle::get_charge(electrons)")
    df = df.Define("electrons_no", "FCCAnalyses::ReconstructedParticle::get_n(electrons)")


    # lepton kinematic histograms
    results.append(df.Histo1D(("muons_all_p_cut0", "", *bins_p), "muons_all_p"))
    results.append(df.Histo1D(("muons_all_theta_cut0", "", *bins_theta), "muons_all_theta"))
    results.append(df.Histo1D(("muons_all_phi_cut0", "", *bins_phi), "muons_all_phi"))
    results.append(df.Histo1D(("muons_all_q_cut0", "", *bins_charge), "muons_all_q"))
    results.append(df.Histo1D(("muons_all_no_cut0", "", *bins_count), "muons_all_no"))

    results.append(df.Histo1D(("electrons_all_p_cut0", "", *bins_p), "electrons_all_p"))
    results.append(df.Histo1D(("electrons_all_theta_cut0", "", *bins_theta), "electrons_all_theta"))
    results.append(df.Histo1D(("electrons_all_phi_cut0", "", *bins_phi), "electrons_all_phi"))
    results.append(df.Histo1D(("electrons_all_q_cut0", "", *bins_charge), "electrons_all_q"))
    results.append(df.Histo1D(("electrons_all_no_cut0", "", *bins_count), "electrons_all_no"))

    
    #########
    ### CUT 0: all events
    #########
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut0"))

    #########
    ### CUT 1: veto muons and electrons
    #########
    df = df.Filter("muons_no == 0")
    df = df.Filter("electrons_no == 0")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut1"))


    ####
    ## CUT 2: missing energy
    ####
    df = df.Define("missingEnergy_rp", "FCCAnalyses::missingEnergy(240., ReconstructedParticles)")
    df = df.Define("missingEnergy_rp_tlv", "FCCAnalyses::makeLorentzVectors(missingEnergy_rp)")
    df = df.Define("missingEnergy", "missingEnergy_rp[0].energy")
    results.append(df.Histo1D(("missingEnergy_nOne", "", *bins_m), "missingEnergy"))
    df = df.Filter("missingEnergy < 30")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut2"))



    #Flavour tagging and jet clustering
    df = jetClusteringHelper4.define(df)
    df = jetFlavourHelper.define_and_inference(df)
    df = df.Define("jet_tlv", "FCCAnalyses::makeLorentzVectors(jet_px, jet_py, jet_pz, jet_e)")
    
    df = df.Filter("jet_tlv.size() >= 4")

    # pair jets based on distance to Z and H masses
    #I copied this from Jake because idk how to do it otherwise.
    df = df.Define("jet_indx", """
            FCCAnalyses::Vec_i min{0, 0, 0, 0};
            float distm = INFINITY;
            for (int i = 0; i < 3; i++)
                for (int j = i + 1; j < 4; j++)
                    for (int k = 0; k < 3; k++) {
                        if (i == k || j == k) continue;
                        for (int l = k + 1; l < 4; l++) {
                            if (i == l || j == l) continue;
                            float distz = (jet_tlv[i] + jet_tlv[j]).M() - 91.2;
                            float disth = (jet_tlv[k] + jet_tlv[l]).M() - 125;
                            if (distz*distz + disth*disth < distm) {
                                distm = distz*distz + disth*disth;
                                min[0] = i; min[1] = j; min[2] = k; min[3] = l;
                            }
                        }
                    }
            return min;""")
  
    df = df.Define("reco_h_jet", "jet_tlv[jet_indx[0]]+jet_tlv[jet_indx[1]]")
    df = df.Define("reco_z_jet", "jet_tlv[jet_indx[2]]+jet_tlv[jet_indx[3]]")

    ##add z and h momentum cuts
    df = df.Define("z_p_reco", "reco_z_jet.P()")
    df = df.Define("h_p_reco", "reco_h_jet.P()")
    results.append(("z_p_reco", "", *bins_p), z_p_reco)
    results.append(("h_p_reco", "", *bins_p), h_p_reco)
  
    #####
    ## CUT 3: Make cut on Z mass
    #####
    df = df.Define("z_m_reco", "reco_z_jet.M()")
    results.append(df.Histo1D(("z_m_reco", "", *bins_m), "z_m_reco"))
    df = df.Filter("z_m_reco < 100 && z_m_reco > 85")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut3"))

    #####
    ## CUT 4: Make cut on H mass
    #####
    df = df.Define("h_m_reco", "reco_h_jet.M()")
    results.append(df.Histo1D(("h_m_reco", "", *bins_m), "h_m_reco"))
    df = df.Filter("h_m_reco < 130 && h_m_reco > 115")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut4"))


    #####
    ## CUT 5: Make cut on B-quark probabilities
    #####
    df = df.Define("recojet_isB_jet0", "recojet_isB[jet_indx[0]]")
    df = df.Define("recojet_isB_jet1", "recojet_isB[jet_indx[1]]")
    results.append(df.Histo1D(("recojet_isB_jet0", "", *bins_score), "recojet_isB_jet0"))
    results.append(df.Histo1D(("recojet_isB_jet1", "", *bins_score), "recojet_isB_jet1"))
    df = df.Filter("recojet_isB_jet0 > 0.95 && recojet_isB_jet1 > 0.95")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut5"))
    
    results.append(df.Histo1D(("jet_p", "", *bins_p), "jet_p"))
    results.append(df.Histo1D(("jet_nconst", "", *(200, 0, 200)), "jet_nconst"))

    ##Add what the z mass decayed into

    return results, weightsum


if __name__ == "__main__":

    datadict = functions.get_datadicts() # get default datasets

    datasets_sig = ["wzp6_ee_nunuH_Hbb_ecm240", "wzp6_ee_mumuH_Hbb_ecm240", "wzp6_ee_tautauH_Hbb_ecm240", "wzp6_ee_ccH_Hbb_ecm240", "wzp6_ee_eeH_Hbb_ecm240", "wzp6_ee_qqH_Hbb_ecm240", "wzp6_ee_ssH_Hbb_ecm240", "wzp6_ee_bbH_Hbb_ecm240"]
    datasets_bkg = ["p8_ee_WW_ecm240", "p8_ee_ZZ_ecm240", "wzp6_ee_eeH_Hcc_ecm240", "wzp6_ee_eeH_Hss_ecm240"]
    datasets_to_run = datasets_sig + datasets_bkg

    functions.build_and_run(datadict, datasets_to_run, build_graph, f"output_h_bb_ee_40GeV_flavourtag.root", args, norm=True, lumi=7200000)
