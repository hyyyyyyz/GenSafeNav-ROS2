import json
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String


# Map PointField datatype constants to (numpy dtype string, byte size)
_FIELD_DTYPE = {
    PointField.INT8:    ('i1', 1),
    PointField.UINT8:   ('u1', 1),
    PointField.INT16:   ('i2', 2),
    PointField.UINT16:  ('u2', 2),
    PointField.INT32:   ('i4', 4),
    PointField.UINT32:  ('u4', 4),
    PointField.FLOAT32: ('f4', 4),
    PointField.FLOAT64: ('f8', 8),
}


def _cloud_to_arrays(msg: PointCloud2):
    """
    Parse a PointCloud2 message into a dict of numpy arrays keyed by field name.
    Returns (arrays: dict[str, np.ndarray], n_points: int).
    """
    n_points = msg.width * msg.height
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n_points, msg.point_step)

    arrays = {}
    for f in msg.fields:
        dtype_str, size = _FIELD_DTYPE.get(f.datatype, ('f4', 4))
        # Copy the slice for each field so the array is writable
        chunk = raw[:, f.offset: f.offset + size].copy()
        arrays[f.name] = np.frombuffer(chunk.tobytes(), dtype=np.dtype(dtype_str))

    return arrays, n_points


def _arrays_to_cloud(header, original_fields, arrays, extra_fields, n_points) -> PointCloud2:
    """
    Build a PointCloud2 from existing fields + extra float32 fields.

    Parameters
    ----------
    header          : original message header (passed through)
    original_fields : list of PointField from the incoming cloud
    arrays          : dict[str, np.ndarray] containing all field data
    extra_fields    : list of str, names of additional float32 fields to append
    n_points        : number of points
    """
    # Rebuild PointField list (original + new)
    new_fields = list(original_fields)
    offset = max(f.offset + _FIELD_DTYPE.get(f.datatype, ('f4', 4))[1]
                 for f in original_fields)
    for name in extra_fields:
        pf = PointField()
        pf.name = name
        pf.offset = offset
        pf.datatype = PointField.FLOAT32
        pf.count = 1
        new_fields.append(pf)
        offset += 4  # float32 = 4 bytes

    new_point_step = offset

    # Allocate output buffer
    buf = np.zeros((n_points, new_point_step), dtype=np.uint8)

    # Copy original fields into the new buffer
    for f in original_fields:
        _, size = _FIELD_DTYPE.get(f.datatype, ('f4', 4))
        src = arrays[f.name].view(np.uint8).reshape(n_points, size)
        buf[:, f.offset: f.offset + size] = src

    # Write extra float32 fields
    for pf in new_fields:
        if pf.name in extra_fields:
            src = arrays[pf.name].astype(np.float32).view(np.uint8).reshape(n_points, 4)
            buf[:, pf.offset: pf.offset + 4] = src

    out = PointCloud2()
    out.header = header
    out.height = 1
    out.width = n_points
    out.fields = new_fields
    out.is_bigendian = False
    out.point_step = new_point_step
    out.row_step = new_point_step * n_points
    out.data = buf.tobytes()
    out.is_dense = False
    return out


class PointCloudProcessNode(Node):
    """
    Subscribes to the raw LiDAR point cloud and the SORT tracker output.
    For each point that falls within `pedestrian_radius` of a tracked
    pedestrian (judged in the XY plane), the node appends the pedestrian's
    velocity (vx, vy) to that point.  All other points receive (vx=0, vy=0).
    """

    def __init__(self):
        super().__init__('pointcloud_annotator_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('input_cloud_topic',  '/livox/lidar/pointcloud'),
                ('tracks_topic',       '/tracked_objects_json'),
                ('output_cloud_topic', '/processed_pointcloud'),
                ('pedestrian_radius',  0.6),   # metres, half the SORT box_size
                ('cloud_queue_size',   5),
                ('tracks_queue_size',  10),
            ]
        )

        self._input_topic  = self.get_parameter('input_cloud_topic').value
        self._tracks_topic = self.get_parameter('tracks_topic').value
        self._output_topic = self.get_parameter('output_cloud_topic').value
        self._radius       = float(self.get_parameter('pedestrian_radius').value)
        cloud_qs           = self.get_parameter('cloud_queue_size').value
        tracks_qs          = self.get_parameter('tracks_queue_size').value

        # Latest tracked pedestrians: list of {x, y, vx, vy}
        self._tracks = []

        self._pub = self.create_publisher(PointCloud2, self._output_topic, cloud_qs)

        self._tracks_sub = self.create_subscription(
            String, self._tracks_topic, self._tracks_callback, tracks_qs
        )
        self._cloud_sub = self.create_subscription(
            PointCloud2, self._input_topic, self._cloud_callback, cloud_qs
        )

        self.get_logger().info(
            f'[PCAnnotator] Listening on: {self._input_topic} + {self._tracks_topic}\n'
            f'              Publishing:   {self._output_topic}\n'
            f'              Radius:       {self._radius} m'
        )

    # ------------------------------------------------------------------
    def _tracks_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            self._tracks = data.get('tracks', [])
        except json.JSONDecodeError:
            self._tracks = []

    # ------------------------------------------------------------------
    def _cloud_callback(self, msg: PointCloud2):
        if self._pub.get_subscription_count() == 0:
            return

        arrays, n_points = _cloud_to_arrays(msg)

        # Initialise velocity arrays with zeros
        vx_arr = np.zeros(n_points, dtype=np.float32)
        vy_arr = np.zeros(n_points, dtype=np.float32)

        if self._tracks and n_points > 0 and 'x' in arrays and 'y' in arrays:
            px = arrays['x'].astype(np.float32)
            py = arrays['y'].astype(np.float32)
            r2 = self._radius ** 2

            for track in self._tracks:
                tx  = float(track.get('x',  0.0))
                ty  = float(track.get('y',  0.0))
                tvx = float(track.get('vx', 0.0))
                tvy = float(track.get('vy', 0.0))

                dist2 = (px - tx) ** 2 + (py - ty) ** 2
                mask  = dist2 < r2

                # Later tracks overwrite earlier ones for overlapping regions
                vx_arr[mask] = tvx
                vy_arr[mask] = tvy

        arrays['vx'] = vx_arr
        arrays['vy'] = vy_arr

        out_msg = _arrays_to_cloud(
            msg.header,
            msg.fields,
            arrays,
            ['vx', 'vy'],
            n_points
        )
        self._pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudProcessNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('[PCProcess] Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
