#include "mlir-ext/Dialect/CommonIR/Transforms/Passes.h"

#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::triton::tile {
#define GEN_PASS_DEF_COMMONIRINFERLAYOUT
#include "mlir-ext/Dialect/CommonIR/Transforms/Passes.h.inc"
} // namespace mlir::triton::tile

using namespace mlir;
namespace tile = mlir::triton::tile;

namespace {

struct CommonIRInferLayoutPass
    : public tile::impl::CommonIRInferLayoutBase<CommonIRInferLayoutPass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto nd = tile::LayoutAttr::get(&getContext(), tile::Layout::ND);

    module.walk([&](tile::AllocOp op) {
      if (!op->hasAttr("layout"))
        op->setAttr("layout", nd);
    });

    module.walk([&](tile::CopyOp op) {
      if (!op->hasAttr("src_layout"))
        op->setAttr("src_layout", nd);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::tile::createCommonIRInferLayoutPass() {
  return std::make_unique<CommonIRInferLayoutPass>();
}
