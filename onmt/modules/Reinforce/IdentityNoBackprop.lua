--
--  Copyright (c) 2015, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Author: Marc'Aurelio Ranzato <ranzato@fb.com>
--          Sumit Chopra <spchopra@fb.com>
--          Michael Auli <michaelauli@fb.com>
--          Wojciech Zaremba <zaremba@cs.nyu.edu>
--

local IdentityNoBackprop, parent = torch.class('nn.IdentityNoBackprop',
                                                  'nn.Identity')
-- This is like Identity, except that it does not backpropagate gradients w.r.t.
-- input.

function IdentityNoBackprop:updateGradInput(input, gradOutput)
   
   self.gradInput = self.gradInput or gradOutput.new()
   self.gradInput:resizeAs(gradOutput)
   self.gradInput:zero()
   
   return self.gradInput
end
