////////////////////////////////////////////////////////////////////////
// Class:       CORSIKAGen
// Plugin Type: producer (art v2_13_00)
// File:        CORSIKAGen_module.cc
//
// Generated at Thu Jan 30 15:45:47 2020 by Dung Phan using cetskelgen
// from cetlib version v3_06_01.
////////////////////////////////////////////////////////////////////////

#include <sqlite3.h>
#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <iomanip>
#include <vector>

#include "TDatabasePDG.h"
#include "TString.h"
#include "TSystem.h"

// GENIE includes
#include "GENIE/Framework/Numerical/RandomGen.h"
#include "Geometry/Geometry.h"
#include "SummaryData/POTSum.h"
#include "SummaryData/RunData.h"
#include "SummaryData/SpillData.h"
#include "SummaryData/SubRunData.h"
#include "TStopwatch.h"
#include "Utilities/AssociationUtil.h"
#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/SubRun.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/ParameterSet.h"
#include "ifdh.h"  //to handle flux files
#include "nusimdata/SimulationBase/GTruth.h"
#include "nusimdata/SimulationBase/MCParticle.h"
#include "nusimdata/SimulationBase/MCTruth.h"

namespace evgen {
class CORSIKAGen;
}

class evgen::CORSIKAGen : public art::EDProducer {
 public:
  explicit CORSIKAGen(fhicl::ParameterSet const &p);
  virtual ~CORSIKAGen();

  void produce(art::Event &evt) override;
  virtual void beginRun(art::Run &run);
  virtual void beginSubRun(art::SubRun& run);
  virtual void endSubRun(art::SubRun& run);

 private:
  void openDBs();
  void populateNShowers();
  void populateTOffset();
  void GetSample(simb::MCTruth &);
  double wrapvar(const double var, const double low, const double high);
  double wrapvarBoxNo(const double var, const double low, const double high,
                      int &boxno);

  void DetectorBigBoxCut(simb::MCTruth &truth);

  bool isIntersectTheBox(const double xyz[], const double dxyz[], double xlo,
                         double xhi, double ylo, double yhi, double zlo,
                         double zhi);

  void ProjectToBoxEdge(const double xyz[], const double dxyz[],
                        const double xlo, const double xhi, const double ylo,
                        const double yhi, const double zlo, const double zhi,
                        double xyzout[]);

  TStopwatch stopwatch_;   ///< keep track of how long it takes to run the job
  genie::RandomGen *rnd_;  ///< Random generator from GENIE

  int m_ShowerInputs = 0;  ///< Number of shower inputs to process from
  std::vector<double> m_NShowersPerEvent;  ///< Number of showers to put in each event of
                                          ///< duration m_fcl_SampleTime; one per showerinput
  std::vector<int> m_MaxShowers;  //< Max number of showers to query, one per showerinput
  double m_ShowerBounds[6] = { 0., 0.,
                              0., 0.,
                              0., 0.};  ///< Boundaries of area over which showers are to be distributed
                                        ///< (x(min), x(max), _unused_, y, z(min), z(max) )
  double m_Toffset_corsika = 0.;  ///< Timing offset to account for propagation time through
                                 ///< atmosphere, populated from db (optional) flux file handling
  ifdh_ns::ifdh *m_IFDH = 0;  ///< (optional) flux file handling
  sqlite3 *fdb[5];           ///< Pointers to sqlite3 database object, max of 5

  // fcl parameters
  double m_fcl_ProjectToHeight = 0.;  ///< Height to which particles will be projected [cm]
  std::vector<std::string> m_fcl_ShowerInputFiles;  ///< Set of CORSIKA shower data files to use
  std::vector<double> m_fcl_ShowerFluxConstants;  ///< Set of flux constants to be associated
                                                 ///< with each shower data file
  double m_fcl_SampleTime = 0.;   ///< Duration of sample [s]
  double m_fcl_Toffset = 0.;  ///< Time offset of sample, defaults to zero (no offset) [s]
  double m_fcl_ShowerAreaExtension = 0.;  ///< Extend distribution of corsika particles in x,z by this much
                                         ///< (e.g. 1000 will extend 10 m in -x, +x, -z, and +z) [cm]
  double m_fcl_RandomXZShift = 0.;  ///< Each shower will be shifted by a random amount in xz so that
                                   ///< showers won't repeatedly sample the same space [cm]
  bool m_fcl_IsBigBoxUsed;  ///< Do we need to use the BigBox cut? The cosmic
                           ///< rays must go through the DetectorBigBox.
                           ///< Otherwise, don't store them

  int m_fcl_Cycle;            ///< MC cycle generation number.
};

evgen::CORSIKAGen::CORSIKAGen(fhicl::ParameterSet const &p)
    : m_fcl_ProjectToHeight(p.get<double>("ProjectToHeight", 0.)),
      m_fcl_ShowerInputFiles(p.get<std::vector<std::string>>("ShowerInputFiles")),
      m_fcl_ShowerFluxConstants(p.get<std::vector<double>>("ShowerFluxConstants")),
      m_fcl_SampleTime(p.get<double>("SampleTime", 0.)),
      m_fcl_Toffset(p.get<double>("TimeOffset", 0.)),
      m_fcl_ShowerAreaExtension(p.get<double>("ShowerAreaExtension", 0.)),
      m_fcl_RandomXZShift(p.get<double>("RandomXZShift", 0.)),
      m_fcl_IsBigBoxUsed(p.get<bool>("IsBigBoxUsed", true)),
      m_fcl_Cycle(p.get<int>("Cycle", 0)) {
  stopwatch_.Start();

  rnd_ = genie::RandomGen::Instance();
  rnd_->SetSeed(std::time(NULL));

  if (m_fcl_ShowerInputFiles.size() != m_fcl_ShowerFluxConstants.size() || m_fcl_ShowerInputFiles.size() == 0 || m_fcl_ShowerFluxConstants.size() == 0) {
    throw cet::exception("CORSIKAGen") << "ShowerInputFiles and ShowerFluxConstants have different or invalid sizes!" << "\n";
  }
  m_ShowerInputs = m_fcl_ShowerInputFiles.size();
  if (m_fcl_SampleTime == 0.) throw cet::exception("CORSIKAGen") << "SampleTime not set!";
  if (m_fcl_ProjectToHeight == 0.) LOG_INFO("CORSIKAGen") << "Using 0. for m_fcl_ProjectToHeight!";

  // create a default random engine; obtain the random seed from
  // NuRandomService, unless overridden in configuration with key "Seed"
  this->openDBs();
  this->populateNShowers();
  this->populateTOffset();
  produces<std::vector<simb::MCTruth>>();
  produces<sumdata::SubRunData, art::InSubRun>();
  produces<sumdata::RunData, art::InRun>();
}

void evgen::CORSIKAGen::ProjectToBoxEdge(const double xyz[],
                                         const double indxyz[],
                                         const double xlo, const double xhi,
                                         const double ylo, const double yhi,
                                         const double zlo, const double zhi,
                                         double xyzout[]) {
  // we want to project backwards, so take mirror of momentum
  const double dxyz[3] = {-indxyz[0], -indxyz[1], -indxyz[2]};
  // Compute the distances to the x/y/z walls
  double dx = 99.E99;
  double dy = 99.E99;
  double dz = 99.E99;
  if (dxyz[0] > 0.0) {
    dx = (xhi - xyz[0]) / dxyz[0];
  } else if (dxyz[0] < 0.0) {
    dx = (xlo - xyz[0]) / dxyz[0];
  }
  if (dxyz[1] > 0.0) {
    dy = (yhi - xyz[1]) / dxyz[1];
  } else if (dxyz[1] < 0.0) {
    dy = (ylo - xyz[1]) / dxyz[1];
  }
  if (dxyz[2] > 0.0) {
    dz = (zhi - xyz[2]) / dxyz[2];
  } else if (dxyz[2] < 0.0) {
    dz = (zlo - xyz[2]) / dxyz[2];
  }
  // Choose the shortest distance
  double d = 0.0;
  if (dx < dy && dx < dz)
    d = dx;
  else if (dy < dz && dy < dx)
    d = dy;
  else if (dz < dx && dz < dy)
    d = dz;
  // Make the step
  for (int i = 0; i < 3; ++i) {
    xyzout[i] = xyz[i] + dxyz[i] * d;
  }
}

evgen::CORSIKAGen::~CORSIKAGen() {
  for (int i = 0; i < m_ShowerInputs; i++) {
    sqlite3_close(fdb[i]);
  }
  // cleanup temp files
  m_IFDH->cleanup();

  stopwatch_.Stop();
  LOG_INFO("CORSIKAGen")
      << "real time to produce file: " << stopwatch_.RealTime();
}

void evgen::CORSIKAGen::beginRun(art::Run &run) {
  art::ServiceHandle<geo::Geometry const> geo;
  std::unique_ptr<sumdata::RunData> runcol(new sumdata::RunData(
      geo->DetId(), geo->FileBaseName(), geo->ExtractGDML()));

  run.put(std::move(runcol));
}

void evgen::CORSIKAGen::beginSubRun(art::SubRun & /*subrun*/) {
}

void evgen::CORSIKAGen::endSubRun(art::SubRun &subrun) {
  // store the cycle information
  std::unique_ptr<sumdata::SubRunData> sd(new sumdata::SubRunData(m_fcl_Cycle));
  subrun.put(std::move(sd));
}

void evgen::CORSIKAGen::openDBs() {
  // choose files based on m_fcl_ShowerInputFiles, copy them with ifdh, open them
  // for c2: statement is unused
  // sqlite3_stmt *statement;
  // setup ifdh object
  if (!m_IFDH) m_IFDH = new ifdh_ns::ifdh;
  const char *ifdh_debug_env = std::getenv("IFDH_DEBUG_LEVEL");
  if (ifdh_debug_env) {
    LOG_INFO("CORSIKAGen") << "IFDH_DEBUG_LEVEL: " << ifdh_debug_env << "\n";
    m_IFDH->set_debug(ifdh_debug_env);
  }
  // get ifdh path for each file in m_fcl_ShowerInputFiles, put into
  // selectedflist if 1 file returned, use that file if >1 file returned,
  // randomly select one file if 0 returned, make exeption for missing files
  std::vector<std::pair<std::string, long>> selectedflist;
  for (int i = 0; i < m_ShowerInputs; i++) {
    if (m_fcl_ShowerInputFiles[i].find("*") == std::string::npos) {
      // if there are no wildcards, don't call findMatchingFiles
      selectedflist.push_back(std::make_pair(m_fcl_ShowerInputFiles[i], 0));
      LOG_INFO("CorsikaGen") << "Selected" << selectedflist.back().first << "\n";
    } else {
      // use findMatchingFiles
      std::vector<std::pair<std::string, long>> flist;
      std::string path(gSystem->DirName(m_fcl_ShowerInputFiles[i].c_str()));
      std::string pattern(gSystem->BaseName(m_fcl_ShowerInputFiles[i].c_str()));
      flist = m_IFDH->findMatchingFiles(path, pattern);
      unsigned int selIndex = -1;
      if (flist.size() == 1) {  // 0th element is the search path:pattern
        selIndex = 0;
      } else if (flist.size() > 1) {
        double randomNumber = rnd_->RndNum().Rndm();
        selIndex = (unsigned int)(randomNumber * (flist.size() - 1) + 0.5);  // rnd with rounding, dont allow picking the 0th element
      } else {
        throw cet::exception("CORSIKAGen") << "No files returned for path:pattern: " << path << ":" << pattern << std::endl;
      }
      selectedflist.push_back(flist[selIndex]);
      LOG_INFO("CorsikaGen")
          << "For " << m_fcl_ShowerInputFiles[i] << ":" << pattern << "\nFound "
          << flist.size() << " candidate files"
          << "\nChoosing file number " << selIndex << "\n"
          << "\nSelected " << selectedflist.back().first << "\n";
    }
  }
  // do the fetching, store local filepaths in locallist
  std::vector<std::string> locallist;
  for (unsigned int i = 0; i < selectedflist.size(); i++) {
    LOG_INFO("CorsikaGen") << "Fetching: " << selectedflist[i].first << " " << selectedflist[i].second << "\n";
    std::string fetchedfile(m_IFDH->fetchInput(selectedflist[i].first));
    LOG_DEBUG("CorsikaGen") << "Fetched; local path: " << fetchedfile;
    locallist.push_back(fetchedfile);
  }
  // open the files in m_fcl_ShowerInputFilesLocalPaths with sqlite3
  for (unsigned int i = 0; i < locallist.size(); i++) {
    // prepare and execute statement to attach db file
    int res = sqlite3_open(locallist[i].c_str(), &fdb[i]);
    if (res != SQLITE_OK)
      throw cet::exception("CORSIKAGen")
          << "Error opening db: (" << locallist[i].c_str() << ") (" << res
          << "): " << sqlite3_errmsg(fdb[i]) << "; memory used:<<"
          << sqlite3_memory_used() << "/" << sqlite3_memory_highwater(0)
          << "\n";
    else
      LOG_INFO("CORSIKAGen") << "Attached db " << locallist[i] << "\n";
  }

  /* A short description of the database file:
  Format: SQLite
  N Table: 3
    - input:
      + runnr           int
      + version         float
      + nshow           int
      + model_high      int
      + model_low       int
      + eslope          float
      + erange_high     float
      + erange_low      float
      + ecuts_hadron    float
      + ecuts_muon      float
      + ecuts_electron  float
      + ecuts_photon    float
    - particles:
      + shower          int
      + pdg             int
      + px              float
      + py              float
      + pz              float
      + x               float
      + z               float
      + t               float
      + e               float
    - showers:
      + id              int
      + nparticles      int

  */
}

double evgen::CORSIKAGen::wrapvar(const double var, const double low, const double high) {
  // wrap variable so that it's always between low and high
  return (var - (high - low) * floor(var / (high - low))) + low;
}

double evgen::CORSIKAGen::wrapvarBoxNo(const double var, const double low, const double high, int &boxno) {
  // wrap variable so that it's always between low and high
  boxno = int(floor(var / (high - low)));
  return (var - (high - low) * floor(var / (high - low))) + low;
}

void evgen::CORSIKAGen::populateTOffset() {
  // populate TOffset_corsika by finding minimum ParticleTime from db file
  sqlite3_stmt *statement;
  const std::string kStatement("select min(t) from particles");
  double t = 0.;
  for (int i = 0; i < m_ShowerInputs; i++) {
    // build and do query to get run min(t) from each db
    if (sqlite3_prepare(fdb[i], kStatement.c_str(), -1, &statement, 0) == SQLITE_OK) {
      int res = 0;
      res = sqlite3_step(statement);
      if (res == SQLITE_ROW) {
        t = sqlite3_column_double(statement, 0);
        LOG_INFO("CORSIKAGen") << "For showers input " << i << " found particles.min(t)=" << t << "\n";
        if (i == 0 || t < m_Toffset_corsika) m_Toffset_corsika = t;
      } else {
        throw cet::exception("CORSIKAGen") << "Unexpected sqlite3_step return value: (" << res << "); " << "ERROR:" << sqlite3_errmsg(fdb[i]) << "\n";
      }
    } else {
      throw cet::exception("CORSIKAGen") << "Error preparing statement: (" << kStatement << "); " << "ERROR:" << sqlite3_errmsg(fdb[i]) << "\n";
    }
  }
  LOG_INFO("CORSIKAGen") << "Found corsika timeoffset [ns]: " << m_Toffset_corsika << "\n";
}

void evgen::CORSIKAGen::populateNShowers() {
  // populate vector of the number of showers per event based on:
  // AREA the showers are being distributed over
  // TIME of the event (m_fcl_SampleTime)
  // flux constants that determine the overall normalizations
  // (m_fcl_ShowerFluxConstants) Energy range over which the sample was generated
  // (ERANGE_*) power spectrum over which the sample was generated (ESLOPE)
  // compute shower area based on the maximal x,z dimensions of cryostat
  // boundaries + m_fcl_ShowerAreaExtension
  art::ServiceHandle<geo::Geometry> geom;

  double xlo_cm = 0;
  double xhi_cm = 0;
  double ylo_cm = 0;
  double yhi_cm = 0;
  double zlo_cm = 0;
  double zhi_cm = 0;

  geom->DetectorBigBox(&xlo_cm, &xhi_cm, &ylo_cm, &yhi_cm, &zlo_cm, &zhi_cm);
  // add on m_fcl_ShowerAreaExtension without being clever
  m_ShowerBounds[0] = xlo_cm - m_fcl_ShowerAreaExtension;
  m_ShowerBounds[1] = xhi_cm + m_fcl_ShowerAreaExtension;
  m_ShowerBounds[4] = zlo_cm - m_fcl_ShowerAreaExtension;
  m_ShowerBounds[5] = zhi_cm + m_fcl_ShowerAreaExtension;
  double showersArea = (m_ShowerBounds[1] / 100 - m_ShowerBounds[0] / 100) * (m_ShowerBounds[5] / 100 - m_ShowerBounds[4] / 100);
  LOG_INFO("CORSIKAGen")
      << "Area extended by : " << m_fcl_ShowerAreaExtension
      << "\nShowers to be distributed betweeen: x=" << m_ShowerBounds[0] << ","
      << m_ShowerBounds[1] << " & z=" << m_ShowerBounds[4] << ","
      << m_ShowerBounds[5]
      << "\nShowers to be distributed with random XZ shift: "
      << m_fcl_RandomXZShift << " cm"
      << "\nShowers to be distributed over area: " << showersArea << " m^2"
      << "\nShowers to be distributed over time: " << m_fcl_SampleTime << " s"
      << "\nShowers to be distributed with time offset: " << m_fcl_Toffset
      << " s"
      << "\nShowers to be distributed at y: " << m_ShowerBounds[3] << " cm";
  // db variables
  sqlite3_stmt *statement;
  const std::string kStatement("select erange_high,erange_low,eslope,nshow from input");
  double upperLimitOfEnergyRange = 0., lowerLimitOfEnergyRange = 0., energySlope = 0., oneMinusGamma = 0., EiToOneMinusGamma = 0., EfToOneMinusGamma = 0.;

  for (int i = 0; i < m_ShowerInputs; i++) {
    // build and do query to get run info from databases
    if (sqlite3_prepare(fdb[i], kStatement.c_str(), -1, &statement, 0) == SQLITE_OK) {
      int res = 0;
      res = sqlite3_step(statement);
      if (res == SQLITE_ROW) {
        upperLimitOfEnergyRange = sqlite3_column_double(statement, 0);
        lowerLimitOfEnergyRange = sqlite3_column_double(statement, 1);
        energySlope = sqlite3_column_double(statement, 2);
        m_MaxShowers.push_back(sqlite3_column_int(statement, 3));
        oneMinusGamma = 1 + energySlope;
        EiToOneMinusGamma = pow(lowerLimitOfEnergyRange, oneMinusGamma);
        EfToOneMinusGamma = pow(upperLimitOfEnergyRange, oneMinusGamma);
        mf::LogVerbatim("CORSIKAGen")
            << "For showers input " << i
            << " found e_hi=" << upperLimitOfEnergyRange
            << ", e_lo=" << lowerLimitOfEnergyRange << ", slope=" << energySlope
            << ", k=" << m_fcl_ShowerFluxConstants[i] << "\n";
      } else {
        throw cet::exception("CORSIKAGen")
            << "Unexpected sqlite3_step return value: (" << res << "); "
            << "ERROR:" << sqlite3_errmsg(fdb[i]) << "\n";
      }
    } else {
      throw cet::exception("CORSIKAGen")
          << "Error preparing statement: (" << kStatement << "); "
          << "ERROR:" << sqlite3_errmsg(fdb[i]) << "\n";
    }
    // this is computed, how?
    double NShowers = (M_PI * showersArea * m_fcl_ShowerFluxConstants[i] * (EfToOneMinusGamma - EiToOneMinusGamma) / oneMinusGamma) * m_fcl_SampleTime;
    m_NShowersPerEvent.push_back(NShowers);
    mf::LogVerbatim("CORSIKAGen") << "For showers input " << i << " the number of showers per event is " << (long int)NShowers << "\n";
  }
}

void evgen::CORSIKAGen::GetSample(simb::MCTruth &mctruth) {
  // for each input, randomly pull m_NShowersPerEvent[i] showers from the
  // Particles table and randomly place them in time (between -m_fcl_SampleTime/2
  // and m_fcl_SampleTime/2) wrap their positions based on the size of the area
  // under consideration based on
  // http://nusoft.fnal.gov/larsoft/doxsvn/html/CRYHelper_8cxx_source.html
  // (Sample) query from sqlite db with select * from particles where shower in
  // (select id from showers ORDER BY substr(id*0.51123124141,length(id)+2) limit
  // 100000) ORDER BY substr(shower*0.51123124141,length(shower)+2); where
  // 0.51123124141 is a random seed to allow randomly selecting rows and should
  // be randomly generated for each query the inner order by is to select
  // randomly from the possible shower id's the outer order by is to make sure
  // the shower numbers are ordered randomly (without this, the showers always
  // come out ordered by shower number and 100000 is the number of showers to be
  // selected at random and needs to be less than the number of showers in the
  // showers table TDatabasePDG is for looking up particle masses
  static TDatabasePDG *pdgt = TDatabasePDG::Instance();

  // db variables
  sqlite3_stmt *statement;
  const TString kStatement(
      "select shower,pdg,px,py,pz,x,z,t,e from particles where shower in "
      "(select id from showers ORDER BY substr(id*%f,length(id)+2) limit %d) "
      "ORDER BY substr(shower*%f,length(shower)+2)");
  // get geometry and figure where to project particles to, based on CRYHelper
  art::ServiceHandle<geo::Geometry const> geom;
  double x1 = 0.;
  double x2 = 0.;
  double y1 = 0.;
  double y2 = 0.;
  double z1 = 0.;
  double z2 = 0.;
  geom->WorldBox(&x1, &x2, &y1, &y2, &z1, &z2);
  // make the world box slightly smaller so that the projection to the edge avoids possible rounding errors later on with Geant4

  double fBoxDelta = 1.e-5;

  x1 += fBoxDelta;
  x2 -= fBoxDelta;
  y1 += fBoxDelta;
  y2 = m_fcl_ProjectToHeight;
  z1 += fBoxDelta;
  z2 -= fBoxDelta;

  // populate mctruth
  int ntotalCtr = 0;   // count number of particles added to mctruth
  int lastShower = 0;  // keep track of last shower id so that t can be randomized on every new shower
  long int nShowerCntr = 0;  // keep track of how many showers are left to be added to mctruth
  int nShowerQry = 0;  // number of showers to query from db
  int shower, pdg;
  double px, py, pz, x, z, tParticleTime, etot, showerTime = 0., showerTimex = 0., showerTimez = 0., showerXOffset = 0., showerZOffset = 0., t;
  for (int i = 0; i < m_ShowerInputs; i++) {
    nShowerCntr = rnd_->RndNum().Poisson(m_NShowersPerEvent[i]);
    LOG_INFO("CORSIKAGEN") << " Shower input " << i << " with mean " << m_NShowersPerEvent[i] << " generating " << nShowerCntr
                           << ". Maximum number of showers in database: " << m_MaxShowers[i];
    while (nShowerCntr > 0) {
      // how many showers should we query?
      if (nShowerCntr > m_MaxShowers[i]) nShowerQry = m_MaxShowers[i];  // take the group size
      else nShowerQry = nShowerCntr;  // take the rest that are needed

      // build and do query to get nshowers
      double thisrnd = rnd_->RndNum().Rndm();  // need a new random number for each query
      TString kthisStatement = TString::Format(kStatement.Data(), thisrnd, nShowerQry, thisrnd);
      LOG_INFO("CORSIKAGen") << "Executing: " << kthisStatement;
      if (sqlite3_prepare(fdb[i], kthisStatement.Data(), -1, &statement, 0) == SQLITE_OK) {
        int res = 0;
        // loop over database rows, pushing particles into mctruth object
        while (1) {
          res = sqlite3_step(statement);
          if (res == SQLITE_ROW) {
            /*
             * Memo columns:
             * [0] shower
             * [1] particle ID (PDG)
             * [2] momentum: x component [GeV/c]
             * [3] momentum: y component [GeV/c]
             * [4] momentum: z component [GeV/c]
             * [5] position: x component [cm]
             * [6] position: z component [cm]
             * [7] time [ns]
             * [8] energy [GeV]
             */
            shower = sqlite3_column_int(statement, 0);
            if (shower != lastShower) {
              // each new shower gets its own random time and position offsets
              showerTime = 1e9 * (rnd_->RndNum().Rndm() * m_fcl_SampleTime);  // converting from s to ns
              showerTimex = 1e9 * (rnd_->RndNum().Rndm() * m_fcl_SampleTime);  // converting from s to ns
              showerTimez = 1e9 * (rnd_->RndNum().Rndm() * m_fcl_SampleTime);  // converting from s to ns
              // and a random offset in both z and x controlled by the
              // m_fcl_RandomXZShift parameter
              showerXOffset = rnd_->RndNum().Rndm() * m_fcl_RandomXZShift - (m_fcl_RandomXZShift / 2);
              showerZOffset = rnd_->RndNum().Rndm() * m_fcl_RandomXZShift - (m_fcl_RandomXZShift / 2);
            }
            pdg = sqlite3_column_int(statement, 1);
            // get mass for this particle
            double m = 0.;  // in GeV
            TParticlePDG *pdgp = pdgt->GetParticle(pdg);
            if (pdgp) m = pdgp->Mass();
            // Note: position/momentum in db have north=-x and west=+z, rotate
            // so that +z is north and +x is west get momentum components
            // To do: Can we use the same orientation for NOvA? My uneducated guess is yes. Check with detsim.
            px = sqlite3_column_double(statement, 4);  // uboone x=Particlez
            py = sqlite3_column_double(statement, 3);
            pz = -sqlite3_column_double(statement, 2);  // uboone z=-Particlex
            etot = sqlite3_column_double(statement, 8);

            // get/calculate position components
            int boxnoX = 0, boxnoZ = 0;
            x = wrapvarBoxNo(sqlite3_column_double(statement, 6) + showerXOffset, m_ShowerBounds[0], m_ShowerBounds[1], boxnoX);
            z = wrapvarBoxNo(-sqlite3_column_double(statement, 5) + showerZOffset, m_ShowerBounds[4], m_ShowerBounds[5], boxnoZ);
            tParticleTime = sqlite3_column_double(statement, 7);  // time offset, includes propagation time from top of atmosphere
            // actual particle time is particle surface arrival time
            //+ shower start time
            //+ global offset (fcl parameter, in s)
            //- propagation time through atmosphere
            //+ boxNo{X,Z} time offset to make grid boxes have different shower times
            t = tParticleTime + showerTime + (1e9 * m_fcl_Toffset) - m_Toffset_corsika + showerTimex * boxnoX + showerTimez * boxnoZ;
            // wrap surface arrival so that it's in the desired time window
            t = wrapvar(t, (1e9 * m_fcl_Toffset), 1e9 * (m_fcl_Toffset + m_fcl_SampleTime));
            simb::MCParticle p(ntotalCtr, pdg, "primary", -200, m, 1);

            // project back to worldvol/m_fcl_ProjectToHeight
            /*
             * This back propagation goes from a point on the upper surface of
             * the cryostat back to the edge of the world, except that that
             * world is cut short by `m_fcl_ProjectToHeight` (`y2`) ceiling.
             * The projection will most often lie on that ceiling, but it may
             * end up instead on one of the side edges of the world, or even
             * outside it.
             */
            double xyzo[3];
            double x0[3] = {x, m_ShowerBounds[3], z};
            double dx[3] = {px, py, pz};
            this->ProjectToBoxEdge(x0, dx, x1, x2, y1, y2, z1, z2, xyzo);
            TLorentzVector pos(xyzo[0], xyzo[1], xyzo[2], t);  // time needs to be in ns to match GENIE, etc
            TLorentzVector mom(px, py, pz, etot);
            p.AddTrajectoryPoint(pos, mom);
            mctruth.Add(p);
            ntotalCtr++;
            lastShower = shower;
          } else if (res == SQLITE_DONE) {
            break;
          } else {
            throw cet::exception("CORSIKAGen") << "Unexpected sqlite3_step return value: (" << res << "); " << "ERROR:" << sqlite3_errmsg(fdb[i]) << "\n";
          }
        }  // end while loop particle query
      } else {
        throw cet::exception("CORSIKAGen") << "Error preparing statement: (" << kthisStatement << "); " << "ERROR:" << sqlite3_errmsg(fdb[i]) << "\n";
      }
      nShowerCntr = nShowerCntr - nShowerQry;
    } // end while loop counting nShowerCntr
  }
}

void evgen::CORSIKAGen::produce(art::Event &evt) {
  std::unique_ptr<std::vector<simb::MCTruth>> truthcol(new std::vector<simb::MCTruth>);

  art::ServiceHandle<geo::Geometry const> geom;

  simb::MCTruth pretruth;
  pretruth.SetOrigin(simb::kCosmicRay);
  GetSample(pretruth);
  LOG_INFO("CORSIKAGen") << "GetSample() number of particles returned: " << pretruth.NParticles() << "\n";
  // for (unsigned int iPart = 0; iPart < pretruth.NParticles(); ++iPart) {
  //   auto particle = pretruth.GetParticle(iPart);
  //   double px = particle.Px() / particle.P();
  //   double py = particle.Py() / particle.P();
  //   double pz = particle.Pz() / particle.P();
  //   double vx = particle.EndX();
  //   double vy = particle.EndY();
  //   double vz = particle.EndZ();
  //   std::cout << "PRETRUTH (" << iPart << "): " << Form("%4.2f, %4.2f, %4.2f, %4.2f, %4.2f, %4.2f", vx, vy, vz, px, py, pz) << std::endl;
  // }

  // DetectorBigBox cut
  if (m_fcl_IsBigBoxUsed) this->DetectorBigBoxCut(pretruth);

  LOG_INFO("CORSIKAGen") << "Number of particles from GetSample() crossing DetectorBigBox: " << pretruth.NParticles() << "\n";
  for (unsigned int iPart = 0; iPart < pretruth.NParticles(); ++iPart) {
    auto particle = pretruth.GetParticle(iPart);
    int pdg = abs(particle.PdgCode());
    double px = particle.Px() / particle.P();
    double py = particle.Py() / particle.P();
    double pz = particle.Pz() / particle.P();
    double vx = particle.EndX();
    double vy = particle.EndY();
    double vz = particle.EndZ();
    double vt = particle.EndT();
    std::cout << Form("PRETRUTH (%05i): ", iPart) << std::fixed << std::setw(10) << pdg << ", "
                                                                << std::setw(12) << std::setprecision(5) << vx << ", "
                                                                << std::setw(12) << std::setprecision(5) << vy << ", "
                                                                << std::setw(12) << std::setprecision(5) << vz << ", "
                                                                << std::setw(12) << std::setprecision(5) << vt/1000. << " us, "
                                                                << std::setw(12) << std::setprecision(5) << px << ", "
                                                                << std::setw(12) << std::setprecision(5) << py << ", "
                                                                << std::setw(12) << std::setprecision(5) << pz << std::endl;
  }
  truthcol->push_back(pretruth);
  evt.put(std::move(truthcol));

  return;
}

////////////////////////////////////////////////////////////////////////
// Method to cut the events not going through the DetectorBigBox,
// which is defined as the box surrounding the DetectorEnclosureBox
// by adding an additional length in each of the dimensions.
// The additional length is defined in detectorbigbox.xml
void evgen::CORSIKAGen::DetectorBigBoxCut(simb::MCTruth &truth) {
  simb::MCTruth mctruth_new;

  double x1 = 0;
  double x2 = 0;
  double y1 = 0;
  double y2 = 0;
  double z1 = 0;
  double z2 = 0;

  art::ServiceHandle<geo::Geometry> geo;
  // Get the ranges of the x, y, and z for the "Detector Volume" that the entire
  // detecotr geometry lives in. If any of the pointers is NULL, the
  // corresponding coordinate is ignored.
  geo->DetectorBigBox(&x1, &x2, &y1, &y2, &z1, &z2);

  // Check for any missing range
  if (x1 == 0 && x2 == 0 && y1 == 0 && y2 == 0 && z1 == 0 && z2 == 0) {
    throw cet::exception("NoGeometryBoxCoords") << "No Geometry Box Coordinates Set\n" << __FILE__ << ":" << __LINE__ << "\n";
    return;
  }

  // Loop over the particles stored by CRY
  int npart = truth.NParticles();
  for (int ipart = 0; ipart < npart; ++ipart) {
    simb::MCParticle particle = truth.GetParticle(ipart);

    double px = particle.Px() / particle.P();
    double py = particle.Py() / particle.P();
    double pz = particle.Pz() / particle.P();

    double vx = particle.EndX();
    double vy = particle.EndY();
    double vz = particle.EndZ();

    // For the near detector, move all the cosmics from their position in
    // the window, straight down, to 1cm above the detector big box. This
    // ensures that a much larger fraction of them pass the IntersectTheBox
    // cut below. Subsequently the ProjectCosmicsToSurface call will
    // backtrack them up to the surface. The flux remains correct throughout
    // this scheme due to the homogeneity of the flux in space and time.
    // It does get the correlations wrong, but the chances of two correlated
    // events making it to the ND are remote anyway.
    if (geo->DetId() == novadaq::cnv::kNEARDET) vy = y2 + 1;

    double xyz[3] = {vx, vy, vz};
    double dxyz[3] = {px, py, pz};

    // If intersecting the DetectorBigBox, add particle
    // currently, using the methods in CosmicsGen code until the geometry methods can be validated
    if (this->isIntersectTheBox(xyz, dxyz, x1, x2, y1, y2, z1, z2)) {
      simb::MCParticle part(particle.TrackId(), particle.PdgCode(), particle.Process(), particle.Mother(), particle.Mass(), particle.StatusCode());
      part.AddTrajectoryPoint(TLorentzVector(vx, vy, vz, particle.T()), particle.Momentum());
      mctruth_new.Add(part);
    }
  }  // end of loop over particles

  // reassign the MCTruth
  truth = mctruth_new;
}

bool evgen::CORSIKAGen::isIntersectTheBox(const double xyz[],
                                          const double dxyz[], double xlo,
                                          double xhi, double ylo, double yhi,
                                          double zlo, double zhi) {
  // If inside the box, obviously intesected
  if (xyz[0] >= xlo && xyz[0] <= xhi && xyz[1] >= ylo && xyz[1] <= yhi && xyz[2] >= zlo && xyz[2] <= zhi)
    return true;

  // So, now we know that the particle is outside the box
  double dx = 0., dy = 0., dz = 0.;

  // Checking intersection with 6 planes
  // 1. Check intersection with the upper plane
  dy = xyz[1] - yhi;
  // Check whether the track going from above and down or from below and up.
  // Otherwise not going to intersect this plane
  if ((dy > 0 && dxyz[1] < 0) || (dy < 0 && dxyz[1] > 0)) {
    double dl = fabs(dy / dxyz[1]);
    double x = xyz[0] + dxyz[0] * dl;
    double z = xyz[2] + dxyz[2] * dl;

    // is it inside the square?
    if (x >= xlo && x <= xhi && z >= zlo && z <= zhi) return true;
  }

  // 2. Check intersection with the lower plane
  dy = xyz[1] - ylo;
  // Check whether the track going from above and down or from below and up.
  // Otherwise not going to intersect this plane
  if ((dy > 0 && dxyz[1] < 0) || (dy < 0 && dxyz[1] > 0)) {
    double dl = fabs(dy / dxyz[1]);
    double x = xyz[0] + dxyz[0] * dl;
    double z = xyz[2] + dxyz[2] * dl;

    // is it inside the square?
    if (x >= xlo && x <= xhi && z >= zlo && z <= zhi) return true;
  }

  // 3. Check intersection with the right plane
  dz = xyz[2] - zhi;
  // Check whether the track going from right part to the left or from left part
  // to right. Otherwise not going to intersect this plane
  if ((dz > 0 && dxyz[2] < 0) || (dz < 0 && dxyz[2] > 0)) {
    double dl = fabs(dz / dxyz[2]);
    double x = xyz[0] + dxyz[0] * dl;
    double y = xyz[1] + dxyz[1] * dl;

    // is it inside the square?
    if (x >= xlo && x <= xhi && y >= ylo && y <= yhi) return true;
  }

  // 4. Check intersection with the left plane
  dz = xyz[2] - zlo;
  // Check whether the track going from right part to the left or from left part
  // to right. Otherwise not going to intersect this plane
  if ((dz > 0 && dxyz[2] < 0) || (dz < 0 && dxyz[2] > 0)) {
    double dl = fabs(dz / dxyz[2]);
    double x = xyz[0] + dxyz[0] * dl;
    double y = xyz[1] + dxyz[1] * dl;

    // is it inside the square?
    if (x >= xlo && x <= xhi && y >= ylo && y <= yhi) return true;
  }

  // 5. Check intersection with the far plane
  dx = xyz[0] - xhi;
  // Check whether the track going from far part toward us or from near part to
  // right. Otherwise not going to intersect this plane
  if ((dx > 0 && dxyz[0] < 0) || (dx < 0 && dxyz[0] > 0)) {
    double dl = fabs(dx / dxyz[0]);
    double y = xyz[1] + dxyz[1] * dl;
    double z = xyz[2] + dxyz[2] * dl;

    // is it inside the square?
    if (z >= zlo && z <= zhi && y >= ylo && y <= yhi) return true;
  }

  // 6. Check intersection with the near plane
  dx = xyz[0] - xlo;
  // Check whether the track going from far part toward us or from near part to
  // right. Otherwise not going to intersect this plane
  if ((dx > 0 && dxyz[0] < 0) || (dx < 0 && dxyz[0] > 0)) {
    double dl = fabs(dx / dxyz[0]);
    double y = xyz[1] + dxyz[1] * dl;
    double z = xyz[2] + dxyz[2] * dl;

    // is it inside the square?
    if (z >= zlo && z <= zhi && y >= ylo && y <= yhi) return true;
  }

  return false;
}

DEFINE_ART_MODULE(evgen::CORSIKAGen)
