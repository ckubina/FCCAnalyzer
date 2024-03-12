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
njets = 2 # number of jets to be clustered
jetClusteringHelper2 = helper_jetclustering.ExclusiveJetClusteringHelper(njets, collection="ReconstructedParticles")
jetFlavourHelper = helper_flavourtagger.JetFlavourHelper(jetClusteringHelper2.jets, jetClusteringHelper2.constituents)
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
    ### CUT 1: veto muons
    #########
    df = df.Filter("muons_no == 0")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut1"))

    #########
    ### CUT 2: at least 2 OS electrons, forming resonance
    #########
    df = df.Filter("electrons_no >= 2")
    df = df.Define("electron1_p", "electrons_p[0]")
    df = df.Define("electron2_p", "electrons_p[1]")
    results.append(df.Histo1D(("electron1_p", "", *bins_p), "electron1_p"))
    results.append(df.Histo1D(("electron2_p", "", *bins_p), "electron2_p"))

    # build the Z resonance based on the available muons. Returns the best muon pair compatible with the Z mass and recoil at 125 GeV
    # technically, it returns a ReconstructedParticleData object with index 0 the di-lepton system (Z), index and 2 the leptons of the pair
    df = df.Define("hbuilder_result", "FCCAnalyses::resonanceBuilder_mass_recoil(91.2, 125, 0, 240, false)(electrons, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)")
    
    df = df.Filter("hbuilder_result.size() > 0")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut2"))

    df = df.Define("zee", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{hbuilder_result[0]}") # the Z
    df = df.Define("zee_tlv", "FCCAnalyses::makeLorentzVectors(zee)")
    df = df.Define("zee_leps", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{hbuilder_result[1],hbuilder_result[2]}") # the electrons
    df = df.Define("zee_leps_tlv", "FCCAnalyses::makeLorentzVectors(zee_leps)")
    df = df.Define("zee_m", "FCCAnalyses::ReconstructedParticle::get_mass(zee)[0]")
    df = df.Define("zee_p", "FCCAnalyses::ReconstructedParticle::get_p(zee)[0]")
    df = df.Define("zee_recoil", "FCCAnalyses::ReconstructedParticle::recoilBuilder(240)(zee)")
    df = df.Define("zee_recoil_m", "FCCAnalyses::ReconstructedParticle::get_mass(zee_recoil)[0]")


    #########
    ### CUT 3: recoil cut (H mass)
    #########
    results.append(df.Histo1D(("ee_recoil_m_nOne", "", *bins_m), "zee_recoil_m"))
    df = df.Filter("zee_recoil_m > 100 && zee_recoil_m < 150")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut3"))

    #####
    ### CUT 4: momentum
    #####
    results.append(df.Histo1D(("ee_p_nOne", "", *bins_p), "zee_p"))
    df = df.Filter("zee_p > 20 && zee_p < 65")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut4"))


    ####
    ## CUT 5: cos theta(miss)
    ####
    df = df.Define("missingEnergy_rp", "FCCAnalyses::missingEnergy(240., ReconstructedParticles)")
    df = df.Define("missingEnergy_rp_tlv", "FCCAnalyses::makeLorentzVectors(missingEnergy_rp)")
    df = df.Define("missingEnergy", "missingEnergy_rp[0].energy")
    df = df.Define("cosTheta_miss", "FCCAnalyses::get_cosTheta_miss(missingEnergy_rp)")
    results.append(df.Histo1D(("cosThetaMiss_nOne", "", *bins_cosThetaMiss), "cosTheta_miss"))
    df = df.Filter("cosTheta_miss < 0.98")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut5"))


    ####
    ## CUT 6: missing energy
    ####
    results.append(df.Histo1D(("missingEnergy_nOne", "", *bins_m), "missingEnergy"))
    df = df.Filter("missingEnergy < 30")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut6"))


    ####
    ## CUT 7: acolinearity
    ####
    df = df.Define("acoplanarity", "FCCAnalyses::acoplanarity(zee_leps)")
    df = df.Define("acolinearity", "FCCAnalyses::acolinearity(zee_leps)")
    results.append(df.Histo1D(("acolinearity_nOne", "", *bins_aco), "acolinearity"))
    
    df = df.Filter("acolinearity > 0.05")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut7"))


    #########
    ### CUT 8 :cut on Z mass
    #########
    results.append(df.Histo1D(("zee_m_nOne", "", *bins_m), "zee_m"))
    df = df.Filter("zee_m > 80 && zee_m < 100")
    results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut8"))

    results.append(df.Histo1D(("zee_m", "", *bins_m_zoom), "zee_m"))

    results.append(df.Histo1D(("muons_no", "", *bins_count), "muons_no"))
    results.append(df.Histo1D(("electrons_no", "", *bins_count), "electrons_no"))

    results.append(df.Histo1D(("acoplanarity", "", *bins_aco), "acoplanarity"))
    results.append(df.Histo1D(("acolinearity", "", *bins_aco), "acolinearity"))


    # define PF candidates collection by removing the electrons
    df = df.Define("rps_no_electrons", "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, electrons)")
    # df = df.Define("RP_px", "FCCAnalyses::ReconstructedParticle::get_px(rps_no_electrons)")
    # df = df.Define("RP_py", "FCCAnalyses::ReconstructedParticle::get_py(rps_no_electrons)")
    # df = df.Define("RP_pz","FCCAnalyses::ReconstructedParticle::get_pz(rps_no_electrons)")
    # df = df.Define("RP_e", "FCCAnalyses::ReconstructedParticle::get_e(rps_no_electrons)")
    # df = df.Define("RP_m", "FCCAnalyses::ReconstructedParticle::get_mass(rps_no_electrons)")
    # df = df.Define("RP_q", "FCCAnalyses::ReconstructedParticle::get_charge(rps_no_electrons)")
    # df = df.Define("pseudo_jets", "FCCAnalyses::JetClusteringUtils::set_pseudoJets(RP_px, RP_py, RP_pz, RP_e)")

    #coment out old jet clustering
    # df = df.Define("clustered_jets", "JetClustering::clustering_ee_kt(2, 2, 1, 0)(pseudo_jets)")
    # df = df.Define("jets", "FCCAnalyses::JetClusteringUtils::get_pseudoJets(clustered_jets)")
    # df = df.Define("jetconstituents", "FCCAnalyses::JetClusteringUtils::get_constituents(clustered_jets)")
    # df = df.Define("jets_e", "FCCAnalyses::JetClusteringUtils::get_e(jets)")
    # df = df.Define("jets_px", "FCCAnalyses::JetClusteringUtils::get_px(jets)")
    # df = df.Define("jets_py", "FCCAnalyses::JetClusteringUtils::get_py(jets)")
    # df = df.Define("jets_pz", "FCCAnalyses::JetClusteringUtils::get_pz(jets)")
    # df = df.Define("jets_m", "FCCAnalyses::JetClusteringUtils::get_m(jets)")

    #Flavour tagging and jet clustering
    df = jetClusteringHelper2.define(df)
    df = jetFlavourHelper.define_and_inference(df)
    df = df.Define("jet_tlv", "FCCAnalyses::makeLorentzVectors(jet_px, jet_py, jet_pz, jet_e)")
    
    if ("jet_tlv.size() < 2"):
        if ("jet_tlv.size() == 1"):
            print("Jet size is 0.")
        else:
            print("Jet size is 1.")
    else:  
        results.append(df.Histo1D(("jet_p", "", *bins_p), "jet_p"))
        results.append(df.Histo1D(("jet_nconst", "", *(200, 0, 200)), "jet_nconst"))
    
        #get probabilities
        df = df.Define("recojet_isB_jet0", "recojet_isB[0]")
        df = df.Define("recojet_isB_jet1", "recojet_isB[1]")
    
        #########
        ### CUT 9 :cut on B quark probabilities
        #########
    
        #Make Graphs
        results.append(df.Histo1D(("recojet_isB_jet0", "", *bins_score), "recojet_isB_jet0"))
        results.append(df.Histo1D(("recojet_isB_jet1", "", *bins_score), "recojet_isB_jet1"))
    
        df = df.Filter("recojet_isB_jet0 > 0.95 && recojet_isB_jet1 > 0.95")
        results.append(df.Histo1D(("cutFlow", "", *bins_count), "cut9"))
    
    
        #Do direct Higgs Mass reconstruction
        df = df.Define("jet0", "jet_tlv[0]")
        df = df.Define("jet1", "jet_tlv[1]")
        df = df.Define("jet0_mass", "jet0.M()")
        df = df.Define("jet1_mass", "jet1.M()")
        df = df.Define("jet0_p", "jet0.P()")
        df = df.Define("jet1_p", "jet1.P()")
        # df = df.Define("dijet_higgs_m_reco", "dijet.M()")
        results.append(df.Histo1D(("jet0_mass", "", *bins_p), "jet0_mass")) #reconstructed higgs mass
        results.append(df.Histo1D(("jet1_mass", "", *bins_p), "jet1_mass"))
        results.append(df.Histo1D(("jet0_p", "", *bins_p), "jet0_p")) #reconstructed higgs mass
        results.append(df.Histo1D(("jet1_p", "", *bins_p), "jet1_p"))

    # compare with jet-truth analysis
    # df = df.Define("jets_mc", "FCCAnalyses::jetTruthFinder(_jetc, ReconstructedParticles, Particle, MCRecoAssociations1)")
    # df = df.Define("njets", f"{njets}")
    # df = df.Define("jets_higgs_mc", "FCCAnalyses::Vec_i res; for(int i=0;i<njets;i++) if(abs(jets_mc[i])==5) res.push_back(i); return res;") # assume H->bb
    # df = df.Filter("jets_higgs_mc.size()==2")
    # df = df.Define("dijet_higgs_m_mc", "(jet_tlv[jets_higgs_mc[0]]+jet_tlv[jets_higgs_mc[1]]).M()")

    #results.append(df.Histo1D(("dijet_higgs_m_mc", "", *bins_p), "dijet_higgs_m_mc")) #monte carlo higgs mass

    return results, weightsum


if __name__ == "__main__":

    datadict = functions.get_datadicts() # get default datasets

    datasets_sig = ["wzp6_ee_nunuH_Hbb_ecm240", "wzp6_ee_mumuH_Hbb_ecm240", "wzp6_ee_tautauH_Hbb_ecm240", "wzp6_ee_ccH_Hbb_ecm240", "wzp6_ee_eeH_Hbb_ecm240", "wzp6_ee_qqH_Hbb_ecm240", "wzp6_ee_ssH_Hbb_ecm240", "wzp6_ee_bbH_Hbb_ecm240"]
    datasets_bkg = ["p8_ee_WW_ecm240", "p8_ee_ZZ_ecm240"]
    datasets_to_run = datasets_sig + datasets_bkg

    functions.build_and_run(datadict, datasets_to_run, build_graph, f"output_h_bb_ee_40GeV_flavourtag.root", args, norm=True, lumi=7200000)
