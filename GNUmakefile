#
# CorsikaGen build
# dphan@utexas.edu
#

LIB_TYPE    := shared
LIB         := lib$(PACKAGE)CORSIKAGen
LIBCXXFILES := $(wildcard *.cxx)
JOBFILES    := $(wildcard *.fcl)

LIBLINK    := -L$(SRT_PRIVATE_CONTEXT)/lib/$(SRT_SUBDIR) -L$(SRT_PUBLIC_CONTEXT)/lib/$(SRT_SUBDIR) -l$(PACKAGE) -l$(PACKAGE)GENIE

########################################################################
include SoftRelTools/standard.mk
include SoftRelTools/arch_spec_nutools.mk
include SoftRelTools/arch_spec_nugen.mk
include SoftRelTools/arch_spec_root.mk
include SoftRelTools/arch_spec_dk2nu.mk
include SoftRelTools/arch_spec_novadaq.mk
include SoftRelTools/arch_spec_art.mk
include SoftRelTools/arch_spec_genie.mk
include SoftRelTools/arch_spec_ifdhc.mk
include SoftRelTools/arch_spec_ifdhart.mk
include SoftRelTools/arch_spec_nutools.mk

override LIBLIBS += $(LOADLIBES)  -L$(SRT_PRIVATE_CONTEXT)/lib/$(SRT_SUBDIR) -L$(SRT_PUBLIC_CONTEXT)/lib/$(SRT_SUBDIR) -lUtilities -lGeometry -lGeometryObjects -L$(NUTOOLS_LIB) -L$(NUGEN_LIB) -lnusimdata_SimulationBase -L$(DK2NUDATA_LIB) -ldk2nuTree -L$(DK2NUGENIE_LIB)  -lSummaryData -lnugen_NuReweight_art -L$(IFDHC_FQ_DIR)/lib/ -lifdh -lnugen_EventGeneratorBase_GENIE