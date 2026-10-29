import test from "node:test";
import assert from "node:assert/strict";
import { before, joints, alignedError } from "./data.js";
test("past-only samples never read future state", () => {
  assert.equal(before([{ t: 1 }, { t: 2 }], 0.9), null);
  assert.equal(before([{ t: 1 }, { t: 2 }], 1.5).t, 1);
});
test("16D mapping excludes grippers", () => {
  assert.deepEqual(
    joints(
      { values: { state: Array.from({ length: 16 }, (_, i) => i) } },
      "right",
    ),
    [8, 9, 10, 11, 12, 13, 14],
  );
});
test("tracking error excludes stale command gaps", () => {
  const feedback = [
    { t: 1.1, values: { joint_position_left: [1] } },
    { t: 2, values: { joint_position_left: [1] } },
  ];
  assert.deepEqual(
    alignedError(
      feedback,
      [{ t: 1, values: { joint_position_left: [2] } }],
      "left",
      0,
    ),
    [[1.1, 1]],
  );
});
