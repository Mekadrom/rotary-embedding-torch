from rotary_embedding_torch import rotary_embedding_torch, apply_rotary_emb, apply_learned_rotations, broadcat

import unittest
import torch
import torch.testing as tt

class TestRotaryEmbeddingTorch(unittest.TestCase):
    def setUp(self):
        self.dim = 8
        self.theta = 10000
        self.theta *= 1. ** (self.dim / (self.dim - 2))
        self.freqs = 1. / (10000 ** (torch.arange(0, self.dim, 2)[:(self.dim // 2)] / self.dim))

        self.rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim)
        self.xpos_rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim, use_xpos=True)
        self.pixel_rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim, freqs_for='pixel')
        self.learned_freq_rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim, learned_freq=True)

        self.queries_1_4_8 = torch.tensor([
            [[0., 1., 2., 3., 4., 5., 6., 7.],
             [1., 2., 3., 4., 5., 6., 7., 8.],
             [2., 3., 4., 5., 6., 7., 8., 9.],
             [3., 4., 5., 6., 7., 8., 9., 0.]]
        ])

        self.keys_1_4_8 = torch.tensor([
            [[3., 4., 5., 6., 7., 8., 9., 0.],
             [2., 3., 4., 5., 6., 7., 8., 9.],
             [1., 2., 3., 4., 5., 6., 7., 8.],
             [0., 1., 2., 3., 4., 5., 6., 7.]],
        ])

        self.queries_4_8 = self.queries_1_4_8[0]

        self.queries_2_4_8 = torch.tensor([
            [[0., 1., 2., 3., 4., 5., 6., 7.], 
             [1., 2., 3., 4., 5., 6., 7., 8.], 
             [2., 3., 4., 5., 6., 7., 8., 9.], 
             [3., 4., 5., 6., 7., 8., 9., 0.]],
            [[0., 1., 2., 3., 4., 5., 6., 7.], 
             [1., 2., 3., 4., 5., 6., 7., 8.], 
             [2., 3., 4., 5., 6., 7., 8., 9.], 
             [3., 4., 5., 6., 7., 8., 9., 0.]],
        ])

        self.queries_2_4_2 = torch.tensor([
            [[0., 7.], 
             [1., 8.], 
             [2., 9.], 
             [3., 0.]],
            [[0., 7.], 
             [1., 8.], 
             [2., 9.], 
             [3., 0.]],
        ])

        self.rotations = torch.tensor([
            [[0., 7.], 
             [1., 8.], 
             [2., 9.], 
             [3., 0.]],
        ]) # shape (1, 4, 2)

        self.freq_ranges = torch.tensor([0, 1])

    def assertTensorsEqual(self, expected, actual):
        self.assertEqual(expected.size(), actual.size())
        tt.assert_close(expected, actual, msg=f'expected {expected}, but got {actual}')

    def test_broadcat(self):
        actual = broadcat([torch.LongTensor([1, 2, 3, 4]), torch.LongTensor([4, 3, 2, 1])])
        expected = torch.LongTensor([1, 2, 3, 4, 4, 3, 2, 1])
        tt.assert_close(actual, expected, msg=f'expected {expected}, but got {actual}')

        actual = broadcat([torch.LongTensor([[1, 2, 3, 4], [4, 3, 2, 1]]), torch.LongTensor([[5, 6, 7, 8], [8, 7, 6, 5]])])
        expected = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8], [4, 3, 2, 1, 8, 7, 6, 5]])
        tt.assert_close(actual, expected, msg=f'expected {expected}, but got {actual}')

        actual = broadcat([torch.LongTensor([[1, 2, 3, 4], [4, 3, 2, 1]]), torch.LongTensor([[5, 6, 7, 8], [8, 7, 6, 5]])], dim=0)
        expected = torch.LongTensor([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8], [8, 7, 6, 5]])
        tt.assert_close(actual, expected, msg=f'expected {expected}, but got {actual}')

    def test_apply_rotary_emb(self):
        # queries or keys tensor with shape (2, 4, 8) - batch_size 2, sequence length 4, feature dimension 8

        q_or_k = self.queries_2_4_8

        actual = apply_rotary_emb(self.freqs, q_or_k)
        expected = torch.tensor([[[-0.8414709568023682,  0.9950041770935059,  1.9699004888534546,
           3.0019986629486084,  4.0000000000000000,  5.0000000000000000,
           6.0000000000000000,  7.0000000000000000],
         [-1.1426396369934082,  2.0898418426513672,  2.9598507881164551,
           4.0029978752136230,  5.0000000000000000,  6.0000000000000000,
           7.0000000000000000,  8.0000000000000000],
         [-1.4438081979751587,  3.1846792697906494,  3.9498007297515869,
           5.0039978027343750,  6.0000000000000000,  7.0000000000000000,
           8.0000000000000000,  9.0000000000000000],
         [-1.7449767589569092,  4.2795171737670898,  4.9397511482238770,
           6.0049972534179688,  7.0000000000000000,  8.0000000000000000,
           9.0000000000000000,  0.0000000000000000]],
        [[-0.8414709568023682,  0.9950041770935059,  1.9699004888534546,
           3.0019986629486084,  4.0000000000000000,  5.0000000000000000,
           6.0000000000000000,  7.0000000000000000],
         [-1.1426396369934082,  2.0898418426513672,  2.9598507881164551,
           4.0029978752136230,  5.0000000000000000,  6.0000000000000000,
           7.0000000000000000,  8.0000000000000000],
         [-1.4438081979751587,  3.1846792697906494,  3.9498007297515869,
           5.0039978027343750,  6.0000000000000000,  7.0000000000000000,
           8.0000000000000000,  9.0000000000000000],
         [-1.7449767589569092,  4.2795171737670898,  4.9397511482238770,
           6.0049972534179688,  7.0000000000000000,  8.0000000000000000,
           9.0000000000000000,  0.0000000000000000]]])

        self.assertTensorsEqual(expected, actual)

        # adding head dimension changes nothing
        q_or_k = self.queries_2_4_8.unsqueeze(0).repeat(2, 1, 1, 1) # shape (2, 2, 4, 8) - batch_size 2, 2 heads, sequence length 4, feature dimension 8

        actual = apply_rotary_emb(self.freqs, q_or_k)
        expected = expected.unsqueeze(0).repeat(2, 1, 1, 1)

        self.assertTensorsEqual(expected, actual)

        # queries or keys tensor with shape (4, 8) - sequence length 4, feature dimension 8
        q_or_k = self.queries_4_8

        actual = apply_rotary_emb(self.freqs, q_or_k)
        expected = torch.tensor([[-0.8414709568023682,  0.9950041770935059,  1.9699004888534546,
          3.0019986629486084,  4.0000000000000000,  5.0000000000000000,
          6.0000000000000000,  7.0000000000000000],
        [-1.1426396369934082,  2.0898418426513672,  2.9598507881164551,
          4.0029978752136230,  5.0000000000000000,  6.0000000000000000,
          7.0000000000000000,  8.0000000000000000],
        [-1.4438081979751587,  3.1846792697906494,  3.9498007297515869,
          5.0039978027343750,  6.0000000000000000,  7.0000000000000000,
          8.0000000000000000,  9.0000000000000000],
        [-1.7449767589569092,  4.2795171737670898,  4.9397511482238770,
          6.0049972534179688,  7.0000000000000000,  8.0000000000000000,
          9.0000000000000000,  0.0000000000000000]])

        self.assertTensorsEqual(expected, actual)

    def test_apply_rotary_emb_raises(self):
        # queries or keys tensor with shape (2, 4, 2) - batch_size 2, sequence length 4, feature dimension 2
        q_or_k = self.queries_2_4_2

        with self.assertRaises(AssertionError) as context:
            actual = apply_rotary_emb(self.freqs, q_or_k)
        self.assertTrue(f'feature dimension {q_or_k.shape[-1]} is not of sufficient size to rotate in all the positions {self.freqs.shape[-1]}' in str(context.exception))

    def test_apply_rotary_emb_scalar_scale(self):
        # queries or keys tensor with shape (1, 4, 8) - batch_size 1, sequence length 4, feature dimension 8
        q_or_k = self.queries_1_4_8

        actual = apply_rotary_emb(self.freqs, q_or_k, scale=4./3.)

        expected = torch.tensor([[[-1.1219613552093506,  1.3266723155975342,  2.6265342235565186,
           4.0026645660400391,  4.0000000000000000,  5.0000000000000000,
           6.0000000000000000,  7.0000000000000000],
         [-1.5235195159912109,  2.7864558696746826,  3.9464678764343262,
           5.3373312950134277,  5.0000000000000000,  6.0000000000000000,
           7.0000000000000000,  8.0000000000000000],
         [-1.9250775575637817,  4.2462391853332520,  5.2664012908935547,
           6.6719970703125000,  6.0000000000000000,  7.0000000000000000,
           8.0000000000000000,  9.0000000000000000],
         [-2.3266358375549316,  5.7060227394104004,  6.5863351821899414,
           8.0066633224487305,  7.0000000000000000,  8.0000000000000000,
           9.0000000000000000,  0.0000000000000000]]])

        self.assertTensorsEqual(expected, actual)

    def test_apply_rotary_emb_tensor_scale(self):
        q_or_k = self.queries_1_4_8

        # length much match sequence length of queries/keys tensor
        scale = torch.tensor([4./3., 3./2., 2./1., 1./2.], dtype=torch.float16)

        actual = apply_rotary_emb(self.freqs, q_or_k, scale=scale)

        expected = torch.tensor([[[-1.1216874122619629,  1.4925062656402588,  3.9398009777069092,
           1.5009993314743042,  4.0000000000000000,  5.0000000000000000,
           6.0000000000000000,  7.0000000000000000],
         [-1.5231475830078125,  3.1347627639770508,  5.9197015762329102,
           2.0014989376068115,  5.0000000000000000,  6.0000000000000000,
           7.0000000000000000,  8.0000000000000000],
         [-1.9246075153350830,  4.7770195007324219,  7.8996014595031738,
           2.5019989013671875,  6.0000000000000000,  7.0000000000000000,
           8.0000000000000000,  9.0000000000000000],
         [-2.3260679244995117,  6.4192752838134766,  9.8795022964477539,
           3.0024986267089844,  7.0000000000000000,  8.0000000000000000,
           9.0000000000000000,  0.0000000000000000]]])

        self.assertTensorsEqual(expected, actual)

    def test_apply_rotary_emb_seq_dim_first(self):
        q_or_k = self.queries_1_4_8.permute(1, 0, 2) # shape (4, 1, 8) - (seq_len, batch_size, feature_dim)

        actual = apply_rotary_emb(self.freqs, q_or_k, seq_dim=0)

        expected = torch.tensor([[[-0.8414709568023682,  0.9950041770935059,  1.9699004888534546,
           3.0019986629486084,  4.0000000000000000,  5.0000000000000000,
           6.0000000000000000,  7.0000000000000000]],
        [[-1.1426396369934082,  2.0898418426513672,  2.9598507881164551,
           4.0029978752136230,  5.0000000000000000,  6.0000000000000000,
           7.0000000000000000,  8.0000000000000000]],
        [[-1.4438081979751587,  3.1846792697906494,  3.9498007297515869,
           5.0039978027343750,  6.0000000000000000,  7.0000000000000000,
           8.0000000000000000,  9.0000000000000000]],
        [[-1.7449767589569092,  4.2795171737670898,  4.9397511482238770,
           6.0049972534179688,  7.0000000000000000,  8.0000000000000000,
           9.0000000000000000,  0.0000000000000000]]])

        self.assertTensorsEqual(expected, actual)

    def test_apply_rotary_emb_diff_start_index(self):
        q_or_k = self.queries_1_4_8

        actual = apply_rotary_emb(self.freqs, q_or_k, start_index=4) # half way through the feature dimension

        expected = torch.tensor([[[ 0.0000000000000000e+00,  1.0000000000000000e+00,
           2.0000000000000000e+00,  3.0000000000000000e+00,
          -2.0461452007293701e+00,  5.3743543624877930e+00,
           5.9297013282775879e+00,  7.0059967041015625e+00],
         [ 1.0000000000000000e+00,  2.0000000000000000e+00,
           3.0000000000000000e+00,  4.0000000000000000e+00,
          -2.3473141193389893e+00,  6.4691920280456543e+00,
           6.9196515083312988e+00,  8.0069961547851562e+00],
         [ 2.0000000000000000e+00,  3.0000000000000000e+00,
           4.0000000000000000e+00,  5.0000000000000000e+00,
          -2.6484827995300293e+00,  7.5640296936035156e+00,
           7.9096012115478516e+00,  9.0079965591430664e+00],
         [ 3.0000000000000000e+00,  4.0000000000000000e+00,
           5.0000000000000000e+00,  6.0000000000000000e+00,
          -2.9496512413024902e+00,  8.6588668823242188e+00,
           8.9995498657226562e+00,  8.9999996125698090e-03]]])

        self.assertTensorsEqual(expected, actual)

    def test_apply_learned_rotations(self):
        q_or_k = self.queries_1_4_8

        actual = apply_learned_rotations(self.rotations, q_or_k)
        expected = torch.tensor([[[ 0.0000000000000000,  1.0000000000000000, -0.4631552696228027,
           3.5756800174713135,  4.0000000000000000,  5.0000000000000000,
           6.0000000000000000,  7.0000000000000000],
         [-1.1426396369934082,  1.9220756292343140, -4.3939332962036133,
           2.3860745429992676,  5.0000000000000000,  6.0000000000000000,
           7.0000000000000000,  8.0000000000000000],
         [-3.5601859092712402,  0.5701543092727661, -5.7051134109497070,
          -2.9071772098541260,  6.0000000000000000,  7.0000000000000000,
           8.0000000000000000,  9.0000000000000000],
         [-3.5344574451446533, -3.5366101264953613,  5.0000000000000000,
           6.0000000000000000,  7.0000000000000000,  8.0000000000000000,
           9.0000000000000000,  0.0000000000000000]]])

        self.assertTensorsEqual(expected, actual)

    def test_apply_learned_rotations_freq_ranges(self):
        q_or_k = self.queries_1_4_8

        actual = apply_learned_rotations(self.rotations, q_or_k, freq_ranges=self.freq_ranges)
        expected = torch.tensor([[[  0.0000000000000000,   1.0000000000000000,   2.0000000000000000,
            3.0000000000000000,   4.0000000000000000,   5.0000000000000000,
           -0.0754923820495605,   9.2192354202270508],
         [  1.0000000000000000,   2.0000000000000000,  -1.7449767589569092,
            4.6856222152709961,   5.0000000000000000,   6.0000000000000000,
           -8.9333658218383789,   5.7615070343017578],
         [  2.0000000000000000,   3.0000000000000000,  -6.2110743522644043,
            1.5564553737640381,   6.0000000000000000,   7.0000000000000000,
          -10.9981079101562500,  -4.9032244682312012],
         [  3.0000000000000000,   4.0000000000000000,  -5.7966823577880859,
           -5.2343549728393555,   7.0000000000000000,   8.0000000000000000,
            9.0000000000000000,   0.0000000000000000]]])

        self.assertTensorsEqual(expected, actual)

    def test_apply_learned_rotations_start_index(self):
        q_or_k = self.queries_1_4_8

        actual = apply_learned_rotations(self.rotations, q_or_k, start_index=4)
        expected = torch.tensor([[[  0.0000000000000000,   1.0000000000000000,   2.0000000000000000,
            3.0000000000000000,   4.0000000000000000,   5.0000000000000000,
           -0.0754923820495605,   9.2192354202270508],
         [  1.0000000000000000,   2.0000000000000000,   3.0000000000000000,
            4.0000000000000000,  -2.3473141193389893,   7.4491686820983887,
           -8.9333658218383789,   5.7615070343017578],
         [  2.0000000000000000,   3.0000000000000000,   4.0000000000000000,
            5.0000000000000000,  -8.8619632720947266,   2.5427563190460205,
          -10.9981079101562500,  -4.9032244682312012],
         [  3.0000000000000000,   4.0000000000000000,   5.0000000000000000,
            6.0000000000000000,  -8.0589075088500977,  -6.9320998191833496,
            9.0000000000000000,   0.0000000000000000]]])

        self.assertTensorsEqual(expected, actual)

    def test_default_rotary_embedding_rotate_queries_or_keys(self):
        q_or_k = self.queries_1_4_8

        actual = self.rotary_embedding.rotate_queries_or_keys(q_or_k)
        expected = torch.tensor([[[ 0.0000000000000000,  1.0000000000000000,  2.0000000000000000,
           3.0000000000000000,  4.0000000000000000,  5.0000000000000000,
           6.0000000000000000,  7.0000000000000000],
         [-1.1426396369934082,  1.9220756292343140,  2.5856788158416748,
           4.2795171737670898,  4.9397511482238770,  6.0496993064880371,
           6.9919967651367188,  8.0069961547851562],
         [-3.5601859092712402,  0.5701543092727661,  2.9269196987152100,
           5.6950101852416992,  5.8588094711303711,  7.1185917854309082,
           7.9819841384887695,  9.0159816741943359],
         [-3.5344574451446533, -3.5366101264953613,  3.0035610198974609,
           7.2096199989318848,  6.7568864822387695,  8.2063684463500977,
           8.9999599456787109,  0.0269999597221613]]])

        self.assertTensorsEqual(expected, actual)

    def test_different_seq_dim_rotary_embedding_rotate_queries_or_keys(self):
        q_or_k = self.queries_1_4_8.permute(1, 0, 2) # shape (4, 1, 8) - (seq_len, batch_size, feature_dim)

        # seq_dim has to be specified relative to the end of the shape of the tensor. 0 does not work for example, it has to be -3 for a tensor with 3 dimensions.
        actual = self.rotary_embedding.rotate_queries_or_keys(q_or_k, seq_dim=-3)
        expected = torch.tensor([[[ 0.0000000000000000,  1.0000000000000000,  2.0000000000000000,
           3.0000000000000000,  4.0000000000000000,  5.0000000000000000,
           6.0000000000000000,  7.0000000000000000]],
        [[-1.1426396369934082,  1.9220756292343140,  2.5856788158416748,
           4.2795171737670898,  4.9397511482238770,  6.0496993064880371,
           6.9919967651367188,  8.0069961547851562]],
        [[-3.5601859092712402,  0.5701543092727661,  2.9269196987152100,
           5.6950101852416992,  5.8588094711303711,  7.1185917854309082,
           7.9819841384887695,  9.0159816741943359]],
        [[-3.5344574451446533, -3.5366101264953613,  3.0035610198974609,
           7.2096199989318848,  6.7568864822387695,  8.2063684463500977,
           8.9999599456787109,  0.0269999597221613]]])

        self.assertTensorsEqual(expected, actual)

    def test_different_offset_rotary_embedding_rotate_queries_or_keys(self):
        q_or_k = self.queries_1_4_8 # shape (1, 4, 8) - batch_size 1, sequence length 4, feature dimension 8

        actual = self.rotary_embedding.rotate_queries_or_keys(q_or_k, offset=4) # half way through the feature dimension

        expected = torch.tensor([[[ 0.7568024992942810, -0.6536436080932617,  0.6738668680191040,
           3.5420196056365967,  3.7968537807464600,  5.1559576988220215,
           5.9719524383544922,  7.0239443778991699],
         [ 2.2015109062194824, -0.3915998935699463,  0.7150454521179199,
           4.9486069679260254,  4.6938767433166504,  6.2423977851867676,
           6.9599123001098633,  8.0348997116088867],
         [ 2.7585868835449219,  2.3216798305511475,  0.4781301021575928,
           6.3852481842041016,  5.5694556236267090,  7.3471879959106445,
           7.9458560943603516,  9.0478372573852539],
         [-0.3662395477294922,  4.9865689277648926, -0.0410947799682617,
           7.8101415634155273,  6.4233145713806152,  8.4700078964233398,
           8.9997797012329102,  0.0629994869232178]]])

        self.assertTensorsEqual(expected, actual)

    def test_freq_seq_len_ovr_valid_same_len_rotary_embedding_rotate_queries_or_keys(self):
        q_or_k = self.queries_1_4_8 # shape (1, 4, 8) - batch_size 1, sequence length 4, feature dimension 8

        actual = self.rotary_embedding.rotate_queries_or_keys(q_or_k, freq_seq_len=4) # sequence length 4, feature dimension 8

        expected = torch.tensor([[[ 0.0000000000000000,  1.0000000000000000,  2.0000000000000000,
           3.0000000000000000,  4.0000000000000000,  5.0000000000000000,
           6.0000000000000000,  7.0000000000000000],
         [-1.1426396369934082,  1.9220756292343140,  2.5856788158416748,
           4.2795171737670898,  4.9397511482238770,  6.0496993064880371,
           6.9919967651367188,  8.0069961547851562],
         [-3.5601859092712402,  0.5701543092727661,  2.9269196987152100,
           5.6950101852416992,  5.8588094711303711,  7.1185917854309082,
           7.9819841384887695,  9.0159816741943359],
         [-3.5344574451446533, -3.5366101264953613,  3.0035610198974609,
           7.2096199989318848,  6.7568864822387695,  8.2063684463500977,
           8.9999599456787109,  0.0269999597221613]]])

        self.assertTensorsEqual(expected, actual)

    def test_freq_seq_len_ovr_valid_gt_rotary_embedding_rotate_queries_or_keys(self):
        q_or_k = self.queries_1_4_8 # shape (1, 4, 8) - batch_size 1, sequence length 4, feature dimension 8

        actual = self.rotary_embedding.rotate_queries_or_keys(q_or_k, freq_seq_len=8)

        expected = torch.tensor([[[ 0.7568024992942810, -0.6536436080932617,  0.6738668680191040,
           3.5420196056365967,  3.7968537807464600,  5.1559576988220215,
           5.9719524383544922,  7.0239443778991699],
         [ 2.2015109062194824, -0.3915998935699463,  0.7150454521179199,
           4.9486069679260254,  4.6938767433166504,  6.2423977851867676,
           6.9599123001098633,  8.0348997116088867],
         [ 2.7585868835449219,  2.3216798305511475,  0.4781301021575928,
           6.3852481842041016,  5.5694556236267090,  7.3471879959106445,
           7.9458560943603516,  9.0478372573852539],
         [-0.3662395477294922,  4.9865689277648926, -0.0410947799682617,
           7.8101415634155273,  6.4233145713806152,  8.4700078964233398,
           8.9997797012329102,  0.0629994869232178]]])

        self.assertTensorsEqual(expected, actual)

    def test_freq_seq_len_ovr_invalid_lt_rotary_embedding_rotate_queries_or_keys(self):
        q_or_k = self.queries_1_4_8 # shape (1, 4, 8) - batch_size 1, sequence length 4, feature dimension 8

        with self.assertRaises(AssertionError) as context:
            actual = self.rotary_embedding.rotate_queries_or_keys(q_or_k, freq_seq_len=2) # sequence length 2, feature dimension 8

    def test_rotary_embedding_rotate_queries_or_keys_throws_xpos(self):
        q_or_k = self.queries_1_4_8

        with self.assertRaises(AssertionError) as context:
            actual = self.xpos_rotary_embedding.rotate_queries_or_keys(q_or_k)

    def test_rotary_embedding_rotate_queries_with_cached_keys(self):
        queries = self.queries_1_4_8 # shape (1, 4, 8) - batch_size 1, sequence length 4, feature dimension 8
        keys = self.keys_1_4_8 # shape (1, 4, 8) - batch_size 1, sequence length 4, feature dimension 8

        actual_queries, actual_keys = self.rotary_embedding.rotate_queries_with_cached_keys(queries, keys)

        expected_queries = torch.tensor([[[ 0.0000000000000000,  1.0000000000000000,  2.0000000000000000,
           3.0000000000000000,  4.0000000000000000,  5.0000000000000000,
           6.0000000000000000,  7.0000000000000000],
         [-1.1426396369934082,  1.9220756292343140,  2.5856788158416748,
           4.2795171737670898,  4.9397511482238770,  6.0496993064880371,
           6.9919967651367188,  8.0069961547851562],
         [-3.5601859092712402,  0.5701543092727661,  2.9269196987152100,
           5.6950101852416992,  5.8588094711303711,  7.1185917854309082,
           7.9819841384887695,  9.0159816741943359],
         [-3.5344574451446533, -3.5366101264953613,  3.0035610198974609,
           7.2096199989318848,  6.7568864822387695,  8.2063684463500977,
           8.9999599456787109,  0.0269999597221613]]])
        expected_keys = torch.tensor([[[ 3.0000000000000000,  4.0000000000000000,  5.0000000000000000,
           6.0000000000000000,  7.0000000000000000,  8.0000000000000000,
           9.0000000000000000,  0.0000000000000000],
         [-1.4438081979751587,  3.3038489818572998,  3.4808495044708252,
           5.3743543624877930,  5.9297013282775879,  7.0596489906311035,
           7.9909963607788086,  9.0079965591430664],
         [-2.2347416877746582,  0.0770037174224854,  2.1455225944519043,
           4.5162744522094727,  4.8790082931518555,  6.0987935066223145,
           6.9839863777160645,  8.0139846801757812],
         [-0.1411200016736984, -0.9899924993515015,  1.0241123437881470,
           3.4570498466491699,  3.8482227325439453,  5.1177320480346680,
           5.9789733886718750,  7.0179686546325684]]])

        self.assertEqual(expected_queries.size(), actual_queries.size())
        self.assertEqual(expected_keys.size(), actual_keys.size())
        tt.assert_close(actual_queries, expected_queries, msg=f'expected {expected_queries}, but got {actual_queries}')
        tt.assert_close(actual_keys, expected_keys, msg=f'expected {expected_keys}, but got {actual_keys}')

    def test_different_seq_dim_rotary_embedding_rotate_queries_with_cached_keys(self):
        queries = self.queries_1_4_8.permute(1, 0, 2) # shape (4, 1, 8) - (seq_len, batch_size, feature_dim)
        keys = self.keys_1_4_8.permute(1, 0, 2) # shape (4, 1, 8) - (seq_len, batch_size, feature_dim)

        # seq_dim has to be specified relative to the end of the shape of the tensor. 0 does not work for example, it has to be -3 for a tensor with 3 dimensions.
        actual_queries, actual_keys = self.rotary_embedding.rotate_queries_with_cached_keys(queries, keys, seq_dim=-3)

        expected_queries = torch.tensor([[[ 0.0000000000000000,  1.0000000000000000,  2.0000000000000000,
           3.0000000000000000,  4.0000000000000000,  5.0000000000000000,
           6.0000000000000000,  7.0000000000000000]],
        [[-1.1426396369934082,  1.9220756292343140,  2.5856788158416748,
           4.2795171737670898,  4.9397511482238770,  6.0496993064880371,
           6.9919967651367188,  8.0069961547851562]],
        [[-3.5601859092712402,  0.5701543092727661,  2.9269196987152100,
           5.6950101852416992,  5.8588094711303711,  7.1185917854309082,
           7.9819841384887695,  9.0159816741943359]],
        [[-3.5344574451446533, -3.5366101264953613,  3.0035610198974609,
           7.2096199989318848,  6.7568864822387695,  8.2063684463500977,
           8.9999599456787109,  0.0269999597221613]]])
        expected_keys = torch.tensor([[[ 3.0000000000000000,  4.0000000000000000,  5.0000000000000000,
           6.0000000000000000,  7.0000000000000000,  8.0000000000000000,
           9.0000000000000000,  0.0000000000000000]],
        [[-1.4438081979751587,  3.3038489818572998,  3.4808495044708252,
           5.3743543624877930,  5.9297013282775879,  7.0596489906311035,
           7.9909963607788086,  9.0079965591430664]],
        [[-2.2347416877746582,  0.0770037174224854,  2.1455225944519043,
           4.5162744522094727,  4.8790082931518555,  6.0987935066223145,
           6.9839863777160645,  8.0139846801757812]],
        [[-0.1411200016736984, -0.9899924993515015,  1.0241123437881470,
           3.4570498466491699,  3.8482227325439453,  5.1177320480346680,
           5.9789733886718750,  7.0179686546325684]]])

        self.assertEqual(expected_queries.size(), actual_queries.size())
        self.assertEqual(expected_keys.size(), actual_keys.size())
        tt.assert_close(actual_queries, expected_queries, msg=f'expected {expected_queries}, but got {actual_queries}')
        tt.assert_close(actual_keys, expected_keys, msg=f'expected {expected_keys}, but got {actual_keys}')

    def test_rotary_embedding_rotate_queries_or_keys(self):
        queries = self.queries_1_4_8 # shape (1, 4, 8) - batch_size 1, sequence length 4, feature dimension 8
        keys = self.keys_1_4_8 # shape (1, 4, 8) - batch_size 1, sequence length 4, feature dimension 8

        actual_queries, actual_keys = self.xpos_rotary_embedding.rotate_queries_and_keys(queries, keys)

        expected_queries = torch.tensor([[[ 0.0000000000000000,  1.0030015707015991,  2.0034546852111816,
           3.0023059844970703,  4.0196223258972168,  5.0150079727172852,
           6.0103640556335449,  7.0053806304931641],
         [-1.1454386711120605,  1.9249579906463623,  2.5879111289978027,
           4.2811617851257324,  4.9518523216247559,  6.0587716102600098,
           6.9980325698852539,  8.0100736618041992],
         [-3.5601859092712402,  0.5701543092727661,  2.9269196987152100,
           5.6950101852416992,  5.8588094711303711,  7.1185917854309082,
           7.9819841384887695,  9.0159816741943359],
         [-3.5258200168609619, -3.5313138961791992,  3.0009701251983643,
           7.2068500518798828,  6.7403740882873535,  8.1940803527832031,
           8.9921970367431641,  0.0269895885139704]]])
        expected_keys = torch.tensor([[[ 2.9853551387786865,  3.9880297183990479,  4.9913783073425293,
           5.9953918457031250,  6.9658288955688477,  7.9760594367980957,
           8.9844808578491211,  0.0000000000000000],
         [-1.4402798414230347,  3.2989020347595215,  3.4778468608856201,
           5.3722896575927734,  5.9152107238769531,  7.0490779876708984,
           7.9841032028198242,  9.0045347213745117],
         [-2.2347416877746582,  0.0770037174224854,  2.1455225944519043,
           4.5162744522094727,  4.8790082931518555,  6.0987935066223145,
           6.9839863777160645,  8.0139846801757812],
         [-0.1414657086133957, -0.9914771318435669,  1.0249965190887451,
           3.4583785533905029,  3.8576498031616211,  5.1254072189331055,
           5.9841351509094238,  7.0206656455993652]]])

        self.assertEqual(expected_queries.size(), actual_queries.size())
        self.assertEqual(expected_keys.size(), actual_keys.size())
        tt.assert_close(actual_queries, expected_queries, msg=f'expected {expected_queries}, but got {actual_queries}')
        tt.assert_close(actual_keys, expected_keys, msg=f'expected {expected_keys}, but got {actual_keys}')

    def test_rotary_embedding_rotate_queries_and_keys_throws_no_xpos(self):
        queries = self.queries_1_4_8 # shape (1, 4, 8) - batch_size 1, sequence length 4, feature dimension 8
        keys = self.keys_1_4_8 # shape (1, 4, 8) - batch_size 1, sequence length 4, feature dimension 8

        with self.assertRaises(AssertionError) as context:
            actual_queries, actual_keys = self.rotary_embedding.rotate_queries_and_keys(queries, keys)

    def test_different_seq_dim_rotary_embedding_rotate_queries_and_keys(self):
        queries = self.queries_1_4_8.permute(1, 0, 2) # shape (4, 1, 8) - (seq_len, batch_size, feature_dim)
        keys = self.keys_1_4_8.permute(1, 0, 2) # shape (4, 1, 8) - (seq_len, batch_size, feature_dim)

        # seq_dim has to be specified relative to the end of the shape of the tensor. 0 does not work for example, it has to be -3 for a tensor with 3 dimensions.
        actual_queries, actual_keys = self.xpos_rotary_embedding.rotate_queries_and_keys(queries, keys, seq_dim=-3)

        expected_queries = torch.tensor([[[ 0.0000000000000000,  1.0030015707015991,  2.0034546852111816,
           3.0023059844970703,  4.0196223258972168,  5.0150079727172852,
           6.0103640556335449,  7.0053806304931641]],
        [[-1.1454386711120605,  1.9249579906463623,  2.5879111289978027,
           4.2811617851257324,  4.9518523216247559,  6.0587716102600098,
           6.9980325698852539,  8.0100736618041992]],
        [[-3.5601859092712402,  0.5701543092727661,  2.9269196987152100,
           5.6950101852416992,  5.8588094711303711,  7.1185917854309082,
           7.9819841384887695,  9.0159816741943359]],
        [[-3.5258200168609619, -3.5313138961791992,  3.0009701251983643,
           7.2068500518798828,  6.7403740882873535,  8.1940803527832031,
           8.9921970367431641,  0.0269895885139704]]])
        expected_keys = torch.tensor([[[ 2.9853551387786865,  3.9880297183990479,  4.9913783073425293,
           5.9953918457031250,  6.9658288955688477,  7.9760594367980957,
           8.9844808578491211,  0.0000000000000000]],
        [[-1.4402798414230347,  3.2989020347595215,  3.4778468608856201,
           5.3722896575927734,  5.9152107238769531,  7.0490779876708984,
           7.9841032028198242,  9.0045347213745117]],
        [[-2.2347416877746582,  0.0770037174224854,  2.1455225944519043,
           4.5162744522094727,  4.8790082931518555,  6.0987935066223145,
           6.9839863777160645,  8.0139846801757812]],
        [[-0.1414657086133957, -0.9914771318435669,  1.0249965190887451,
           3.4583785533905029,  3.8576498031616211,  5.1254072189331055,
           5.9841351509094238,  7.0206656455993652]]])

        self.assertEqual(expected_queries.size(), actual_queries.size())
        self.assertEqual(expected_keys.size(), actual_keys.size())
        tt.assert_close(actual_queries, expected_queries, msg=f'expected {expected_queries}, but got {actual_queries}')
        tt.assert_close(actual_keys, expected_keys, msg=f'expected {expected_keys}, but got {actual_keys}')

    def test_rotary_embedding_get_scale(self):
        q_or_k = self.queries_1_4_8 # shape (1, 4, 8) - batch_size 1, sequence length 4, feature dimension 8

        seqs = self.xpos_rotary_embedding.get_seq_pos(q_or_k.size(1), q_or_k.device, q_or_k.dtype)

        actual = self.xpos_rotary_embedding.get_scale(seqs)

        expected = torch.tensor([[1.0049055814743042, 1.0030015707015991, 1.0017273426055908,
         1.0007686614990234, 1.0049055814743042, 1.0030015707015991,
         1.0017273426055908, 1.0007686614990234],
        [1.0024497509002686, 1.0014996528625488, 1.0008633136749268,
         1.0003843307495117, 1.0024497509002686, 1.0014996528625488,
         1.0008633136749268, 1.0003843307495117],
        [1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
         1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
         1.0000000000000000, 1.0000000000000000],
        [0.9975562095642090, 0.9985025525093079, 0.9991374015808105,
         0.9996158480644226, 0.9975562095642090, 0.9985025525093079,
         0.9991374015808105, 0.9996158480644226]])

        self.assertTensorsEqual(expected, actual)

    def test_rotary_embedding_get_scale_throws_no_xpos(self):
        q_or_k = self.queries_1_4_8

        seqs = self.rotary_embedding.get_seq_pos(q_or_k.size(1), q_or_k.device, q_or_k.dtype)

        with self.assertRaises(AssertionError) as context:
            actual = self.rotary_embedding.get_scale(seqs)

    def test_different_seq_len_rotary_embedding_get_scale(self):
        q_or_k = self.queries_1_4_8 # shape (1, 4, 8) - batch_size 1, sequence length 4, feature dimension 8

        # setting any explicit sequence length causes the result of get_scale to be cached and returned in subsequent requests with the same sequence length
        seqs = self.xpos_rotary_embedding.get_seq_pos(q_or_k.size(1), q_or_k.device, q_or_k.dtype)

        self.assertTrue(self.xpos_rotary_embedding.cached_scales is None or self.xpos_rotary_embedding.cached_scales.shape[0] == 0)

        actual = self.xpos_rotary_embedding.get_scale(seqs, seq_len=4)

        expected = torch.tensor([[1.0049055814743042, 1.0030015707015991, 1.0017273426055908,
         1.0007686614990234, 1.0049055814743042, 1.0030015707015991,
         1.0017273426055908, 1.0007686614990234],
        [1.0024497509002686, 1.0014996528625488, 1.0008633136749268,
         1.0003843307495117, 1.0024497509002686, 1.0014996528625488,
         1.0008633136749268, 1.0003843307495117],
        [1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
         1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
         1.0000000000000000, 1.0000000000000000],
        [0.9975562095642090, 0.9985025525093079, 0.9991374015808105,
         0.9996158480644226, 0.9975562095642090, 0.9985025525093079,
         0.9991374015808105, 0.9996158480644226]])

        self.assertTensorsEqual(expected, actual)

        self.assertTrue(self.xpos_rotary_embedding.cached_scales.shape[0] >= 4)

        actual = self.xpos_rotary_embedding.get_scale(seqs, seq_len=8)

        # seq_len greater than input seqs tensor doesn't matter, its only used to check against the cache
        self.assertTrue(self.xpos_rotary_embedding.cached_scales.shape[0] < 8)
        self.assertTensorsEqual(expected, actual)

        self.xpos_rotary_embedding.tmp_store('cached_scales', None) # clear the cache

    def test_different_offset_rotary_embedding_get_scale(self):
        q_or_k = self.queries_1_4_8 # shape (1, 4, 8) - batch_size 1, sequence length 4, feature dimension 8

        seqs = self.xpos_rotary_embedding.get_seq_pos(q_or_k.size(1), q_or_k.device, q_or_k.dtype)

        self.assertTrue(self.xpos_rotary_embedding.cached_scales is None or self.xpos_rotary_embedding.cached_scales.shape[0] == 0)

        actual = self.xpos_rotary_embedding.get_scale(seqs, offset=4)

        expected = torch.tensor([[1.0049055814743042, 1.0030015707015991, 1.0017273426055908,
         1.0007686614990234, 1.0049055814743042, 1.0030015707015991,
         1.0017273426055908, 1.0007686614990234],
        [1.0024497509002686, 1.0014996528625488, 1.0008633136749268,
         1.0003843307495117, 1.0024497509002686, 1.0014996528625488,
         1.0008633136749268, 1.0003843307495117],
        [1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
         1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
         1.0000000000000000, 1.0000000000000000],
        [0.9975562095642090, 0.9985025525093079, 0.9991374015808105,
         0.9996158480644226, 0.9975562095642090, 0.9985025525093079,
         0.9991374015808105, 0.9996158480644226]])

        # not cached because seq_len not specified
        self.assertTrue(self.xpos_rotary_embedding.cached_scales is None or self.xpos_rotary_embedding.cached_scales.shape[0] == 0)
        self.assertTensorsEqual(expected, actual)

        q_or_k = self.queries_1_4_8.repeat(1, 2, 1) # shape (1, 8, 8) basically just a different sequence length

        seqs = self.xpos_rotary_embedding.get_seq_pos(q_or_k.size(1), q_or_k.device, q_or_k.dtype)

        # value of seq_len doesn't matter for first call as it only determines whether caching should happen
        actual = self.xpos_rotary_embedding.get_scale(seqs, seq_len=1203912)

        expected = torch.tensor([[1.0098352432250977, 1.0060122013092041, 1.0034577846527100,
         1.0015380382537842, 1.0098352432250977, 1.0060122013092041,
         1.0034577846527100, 1.0015380382537842],
        [1.0073673725128174, 1.0045057535171509, 1.0025922060012817,
         1.0011532306671143, 1.0073673725128174, 1.0045057535171509,
         1.0025922060012817, 1.0011532306671143],
        [1.0049055814743042, 1.0030015707015991, 1.0017273426055908,
         1.0007686614990234, 1.0049055814743042, 1.0030015707015991,
         1.0017273426055908, 1.0007686614990234],
        [1.0024497509002686, 1.0014996528625488, 1.0008633136749268,
         1.0003843307495117, 1.0024497509002686, 1.0014996528625488,
         1.0008633136749268, 1.0003843307495117],
        [1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
         1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
         1.0000000000000000, 1.0000000000000000],
        [0.9975562095642090, 0.9985025525093079, 0.9991374015808105,
         0.9996158480644226, 0.9975562095642090, 0.9985025525093079,
         0.9991374015808105, 0.9996158480644226],
        [0.9951183199882507, 0.9970073699951172, 0.9982755780220032,
         0.9992318749427795, 0.9951183199882507, 0.9970073699951172,
         0.9982755780220032, 0.9992318749427795],
        [0.9926864504814148, 0.9955144524574280, 0.9974144697189331,
         0.9988480806350708, 0.9926864504814148, 0.9955144524574280,
         0.9974144697189331, 0.9988480806350708]])

        # sequence length of 8 is cached after the previous call so check that here
        self.assertTrue(self.xpos_rotary_embedding.cached_scales.shape[0] >= 8)
        self.assertTensorsEqual(expected, actual)

        # value of seq_len does matter for subsequent calls because it checks against the cache's length
        actual = self.xpos_rotary_embedding.get_scale(seqs, seq_len=8)

        expected = torch.tensor([[1.0098352432250977, 1.0060122013092041, 1.0034577846527100,
         1.0015380382537842, 1.0098352432250977, 1.0060122013092041,
         1.0034577846527100, 1.0015380382537842],
        [1.0073673725128174, 1.0045057535171509, 1.0025922060012817,
         1.0011532306671143, 1.0073673725128174, 1.0045057535171509,
         1.0025922060012817, 1.0011532306671143],
        [1.0049055814743042, 1.0030015707015991, 1.0017273426055908,
         1.0007686614990234, 1.0049055814743042, 1.0030015707015991,
         1.0017273426055908, 1.0007686614990234],
        [1.0024497509002686, 1.0014996528625488, 1.0008633136749268,
         1.0003843307495117, 1.0024497509002686, 1.0014996528625488,
         1.0008633136749268, 1.0003843307495117],
        [1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
         1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
         1.0000000000000000, 1.0000000000000000],
        [0.9975562095642090, 0.9985025525093079, 0.9991374015808105,
         0.9996158480644226, 0.9975562095642090, 0.9985025525093079,
         0.9991374015808105, 0.9996158480644226],
        [0.9951183199882507, 0.9970073699951172, 0.9982755780220032,
         0.9992318749427795, 0.9951183199882507, 0.9970073699951172,
         0.9982755780220032, 0.9992318749427795],
        [0.9926864504814148, 0.9955144524574280, 0.9974144697189331,
         0.9988480806350708, 0.9926864504814148, 0.9955144524574280,
         0.9974144697189331, 0.9988480806350708]])
        
        self.assertTensorsEqual(expected, actual)

        self.xpos_rotary_embedding.tmp_store('cached_scales', None) # clear the cache

    def test_rotary_embedding_get_axial_freqs(self):
        actual_no_xpos = self.rotary_embedding.get_axial_freqs(2, 2)
        actual_xpos = self.xpos_rotary_embedding.get_axial_freqs(2, 2)

        # xpos has no bearing on the results of get_axial_freqs
        self.assertEqual(actual_no_xpos.size(), actual_xpos.size())
        tt.assert_close(actual_no_xpos, actual_xpos, msg=f'expected {actual_no_xpos}, but got {actual_xpos}')

        expected = torch.tensor([[[0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
          0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
          0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
          0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
          0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
          0.0000000000000000],
         [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
          0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
          0.0000000000000000, 0.0000000000000000, 1.0000000000000000,
          1.0000000000000000, 0.1000000014901161, 0.1000000014901161,
          0.0099999997764826, 0.0099999997764826, 0.0010000000474975,
          0.0010000000474975]],
        [[1.0000000000000000, 1.0000000000000000, 0.1000000014901161,
          0.1000000014901161, 0.0099999997764826, 0.0099999997764826,
          0.0010000000474975, 0.0010000000474975, 0.0000000000000000,
          0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
          0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
          0.0000000000000000],
         [1.0000000000000000, 1.0000000000000000, 0.1000000014901161,
          0.1000000014901161, 0.0099999997764826, 0.0099999997764826,
          0.0010000000474975, 0.0010000000474975, 1.0000000000000000,
          1.0000000000000000, 0.1000000014901161, 0.1000000014901161,
          0.0099999997764826, 0.0099999997764826, 0.0010000000474975,
          0.0010000000474975]]])
        
        self.assertEqual(expected.size(), actual_no_xpos.size())
        tt.assert_close(actual_no_xpos, expected, msg=f'expected {expected}, but got {actual_no_xpos}')

        actual_no_xpos = self.rotary_embedding.get_axial_freqs(2)

        expected = expected[0][:, -8:] # result is only different slice of the same tensor when there are fewer dimensions passed in

        self.assertEqual(expected.size(), actual_no_xpos.size())
        tt.assert_close(actual_no_xpos, expected, msg=f'expected {expected}, but got {actual_no_xpos}')

        actual_no_xpos = self.rotary_embedding.get_axial_freqs(1)

        expected = expected[:1, :] # result is not repeated twice along first dimension

        tt.assert_close(actual_no_xpos, expected, msg=f'expected {expected}, but got {actual_no_xpos}')
        self.assertEqual(expected.size(), actual_no_xpos.size())

    def test_pixel_rotary_embedding_get_axial_freqs(self):
        actual = self.pixel_rotary_embedding.get_axial_freqs(2, 2)

        expected = torch.tensor([[[ -3.1415927410125732,  -3.1415927410125732,  -7.3303837776184082,
           -7.3303837776184082, -11.5191726684570312, -11.5191726684570312,
          -15.7079639434814453, -15.7079639434814453,  -3.1415927410125732,
           -3.1415927410125732,  -7.3303837776184082,  -7.3303837776184082,
          -11.5191726684570312, -11.5191726684570312, -15.7079639434814453,
          -15.7079639434814453],
         [ -3.1415927410125732,  -3.1415927410125732,  -7.3303837776184082,
           -7.3303837776184082, -11.5191726684570312, -11.5191726684570312,
          -15.7079639434814453, -15.7079639434814453,   3.1415927410125732,
            3.1415927410125732,   7.3303837776184082,   7.3303837776184082,
           11.5191726684570312,  11.5191726684570312,  15.7079639434814453,
           15.7079639434814453]],
        [[  3.1415927410125732,   3.1415927410125732,   7.3303837776184082,
            7.3303837776184082,  11.5191726684570312,  11.5191726684570312,
           15.7079639434814453,  15.7079639434814453,  -3.1415927410125732,
           -3.1415927410125732,  -7.3303837776184082,  -7.3303837776184082,
          -11.5191726684570312, -11.5191726684570312, -15.7079639434814453,
          -15.7079639434814453],
         [  3.1415927410125732,   3.1415927410125732,   7.3303837776184082,
            7.3303837776184082,  11.5191726684570312,  11.5191726684570312,
           15.7079639434814453,  15.7079639434814453,   3.1415927410125732,
            3.1415927410125732,   7.3303837776184082,   7.3303837776184082,
           11.5191726684570312,  11.5191726684570312,  15.7079639434814453,
           15.7079639434814453]]])

        self.assertTensorsEqual(expected, actual)

        actual = self.pixel_rotary_embedding.get_axial_freqs(2)
        expected = expected[0][:, -8:]

        self.assertTensorsEqual(expected, actual)

        actual = self.pixel_rotary_embedding.get_axial_freqs(1)

        expected = expected[:1, :] # result is not repeated twice along first dimension

        self.assertTensorsEqual(expected, actual)

    def test_rotary_embedding_forward(self):
        q_or_k = self.queries_1_4_8 # shape (1, 4, 8) - batch_size 1, sequence length 4, feature dimension 8

        seqs = self.rotary_embedding.get_seq_pos(q_or_k.size(1), q_or_k.device, q_or_k.dtype)

        self.assertTrue(self.rotary_embedding.cached_freqs is None or self.rotary_embedding.cached_freqs.shape[0] == 0)

        actual = self.rotary_embedding(seqs)

        expected = torch.tensor([[0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
         0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
         0.0000000000000000e+00, 0.0000000000000000e+00],
        [1.0000000000000000e+00, 1.0000000000000000e+00, 1.0000000149011612e-01,
         1.0000000149011612e-01, 9.9999997764825821e-03, 9.9999997764825821e-03,
         1.0000000474974513e-03, 1.0000000474974513e-03],
        [2.0000000000000000e+00, 2.0000000000000000e+00, 2.0000000298023224e-01,
         2.0000000298023224e-01, 1.9999999552965164e-02, 1.9999999552965164e-02,
         2.0000000949949026e-03, 2.0000000949949026e-03],
        [3.0000000000000000e+00, 3.0000000000000000e+00, 3.0000001192092896e-01,
         3.0000001192092896e-01, 2.9999999329447746e-02, 2.9999999329447746e-02,
         3.0000000260770321e-03, 3.0000000260770321e-03]])

        self.assertTensorsEqual(expected, actual)

        # check that the scales are not cached
        self.assertTrue(self.rotary_embedding.cached_freqs is None or self.rotary_embedding.cached_freqs.shape[0] == 0)

    def test_different_seq_len_rotary_embedding_forward(self):
        q_or_k = self.queries_1_4_8.repeat(1, 2, 1) # shape (1, 8, 8)

        seqs = self.rotary_embedding.get_seq_pos(q_or_k.size(1), q_or_k.device, q_or_k.dtype)

        self.assertTrue(self.rotary_embedding.cached_freqs is None or self.rotary_embedding.cached_freqs.shape[0] == 0)

        # again, seq_len just needs to be specified for caching to happen initially
        actual = self.rotary_embedding(seqs, seq_len=125125124)

        expected = torch.tensor([[0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
         0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
         0.0000000000000000e+00, 0.0000000000000000e+00],
        [1.0000000000000000e+00, 1.0000000000000000e+00, 1.0000000149011612e-01,
         1.0000000149011612e-01, 9.9999997764825821e-03, 9.9999997764825821e-03,
         1.0000000474974513e-03, 1.0000000474974513e-03],
        [2.0000000000000000e+00, 2.0000000000000000e+00, 2.0000000298023224e-01,
         2.0000000298023224e-01, 1.9999999552965164e-02, 1.9999999552965164e-02,
         2.0000000949949026e-03, 2.0000000949949026e-03],
        [3.0000000000000000e+00, 3.0000000000000000e+00, 3.0000001192092896e-01,
         3.0000001192092896e-01, 2.9999999329447746e-02, 2.9999999329447746e-02,
         3.0000000260770321e-03, 3.0000000260770321e-03],
        [4.0000000000000000e+00, 4.0000000000000000e+00, 4.0000000596046448e-01,
         4.0000000596046448e-01, 3.9999999105930328e-02, 3.9999999105930328e-02,
         4.0000001899898052e-03, 4.0000001899898052e-03],
        [5.0000000000000000e+00, 5.0000000000000000e+00, 5.0000000000000000e-01,
         5.0000000000000000e-01, 4.9999997019767761e-02, 4.9999997019767761e-02,
         5.0000003539025784e-03, 5.0000003539025784e-03],
        [6.0000000000000000e+00, 6.0000000000000000e+00, 6.0000002384185791e-01,
         6.0000002384185791e-01, 5.9999998658895493e-02, 5.9999998658895493e-02,
         6.0000000521540642e-03, 6.0000000521540642e-03],
        [7.0000000000000000e+00, 7.0000000000000000e+00, 6.9999998807907104e-01,
         6.9999998807907104e-01, 7.0000000298023224e-02, 7.0000000298023224e-02,
         7.0000002160668373e-03, 7.0000002160668373e-03]])

        self.assertTrue(self.rotary_embedding.cached_freqs.shape[0] >= 8) # check that the freqs are cached
        self.assertTensorsEqual(expected, actual)

        self.rotary_embedding.tmp_store('cached_freqs', None) # clear the cache

    def test_different_offset_rotary_embedding_forward(self):
        q_or_k = self.queries_1_4_8.repeat(1, 2, 1) # shape (1, 8, 8)

        seqs = self.rotary_embedding.get_seq_pos(q_or_k.size(1), q_or_k.device, q_or_k.dtype)

        self.assertTrue(self.rotary_embedding.cached_freqs is None or self.rotary_embedding.cached_freqs.shape[0] == 0)

        # offset changes nothing if seq_len is not set
        actual = self.rotary_embedding(seqs, offset=125126125)

        expected = torch.tensor([[0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
         0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
         0.0000000000000000e+00, 0.0000000000000000e+00],
        [1.0000000000000000e+00, 1.0000000000000000e+00, 1.0000000149011612e-01,
         1.0000000149011612e-01, 9.9999997764825821e-03, 9.9999997764825821e-03,
         1.0000000474974513e-03, 1.0000000474974513e-03],
        [2.0000000000000000e+00, 2.0000000000000000e+00, 2.0000000298023224e-01,
         2.0000000298023224e-01, 1.9999999552965164e-02, 1.9999999552965164e-02,
         2.0000000949949026e-03, 2.0000000949949026e-03],
        [3.0000000000000000e+00, 3.0000000000000000e+00, 3.0000001192092896e-01,
         3.0000001192092896e-01, 2.9999999329447746e-02, 2.9999999329447746e-02,
         3.0000000260770321e-03, 3.0000000260770321e-03],
        [4.0000000000000000e+00, 4.0000000000000000e+00, 4.0000000596046448e-01,
         4.0000000596046448e-01, 3.9999999105930328e-02, 3.9999999105930328e-02,
         4.0000001899898052e-03, 4.0000001899898052e-03],
        [5.0000000000000000e+00, 5.0000000000000000e+00, 5.0000000000000000e-01,
         5.0000000000000000e-01, 4.9999997019767761e-02, 4.9999997019767761e-02,
         5.0000003539025784e-03, 5.0000003539025784e-03],
        [6.0000000000000000e+00, 6.0000000000000000e+00, 6.0000002384185791e-01,
         6.0000002384185791e-01, 5.9999998658895493e-02, 5.9999998658895493e-02,
         6.0000000521540642e-03, 6.0000000521540642e-03],
        [7.0000000000000000e+00, 7.0000000000000000e+00, 6.9999998807907104e-01,
         6.9999998807907104e-01, 7.0000000298023224e-02, 7.0000000298023224e-02,
         7.0000002160668373e-03, 7.0000002160668373e-03]])

        self.assertTrue(self.rotary_embedding.cached_freqs is None or self.rotary_embedding.cached_freqs.shape[0] == 0)
        self.assertTensorsEqual(expected, actual)

        # causes caching to happen
        actual = self.rotary_embedding(seqs, seq_len=125126126, offset=4)

        expected = torch.tensor([[0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
         0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
         0.0000000000000000e+00, 0.0000000000000000e+00],
        [1.0000000000000000e+00, 1.0000000000000000e+00, 1.0000000149011612e-01,
         1.0000000149011612e-01, 9.9999997764825821e-03, 9.9999997764825821e-03,
         1.0000000474974513e-03, 1.0000000474974513e-03],
        [2.0000000000000000e+00, 2.0000000000000000e+00, 2.0000000298023224e-01,
         2.0000000298023224e-01, 1.9999999552965164e-02, 1.9999999552965164e-02,
         2.0000000949949026e-03, 2.0000000949949026e-03],
        [3.0000000000000000e+00, 3.0000000000000000e+00, 3.0000001192092896e-01,
         3.0000001192092896e-01, 2.9999999329447746e-02, 2.9999999329447746e-02,
         3.0000000260770321e-03, 3.0000000260770321e-03],
        [4.0000000000000000e+00, 4.0000000000000000e+00, 4.0000000596046448e-01,
         4.0000000596046448e-01, 3.9999999105930328e-02, 3.9999999105930328e-02,
         4.0000001899898052e-03, 4.0000001899898052e-03],
        [5.0000000000000000e+00, 5.0000000000000000e+00, 5.0000000000000000e-01,
         5.0000000000000000e-01, 4.9999997019767761e-02, 4.9999997019767761e-02,
         5.0000003539025784e-03, 5.0000003539025784e-03],
        [6.0000000000000000e+00, 6.0000000000000000e+00, 6.0000002384185791e-01,
         6.0000002384185791e-01, 5.9999998658895493e-02, 5.9999998658895493e-02,
         6.0000000521540642e-03, 6.0000000521540642e-03],
        [7.0000000000000000e+00, 7.0000000000000000e+00, 6.9999998807907104e-01,
         6.9999998807907104e-01, 7.0000000298023224e-02, 7.0000000298023224e-02,
         7.0000002160668373e-03, 7.0000002160668373e-03]])

        self.assertTrue(self.rotary_embedding.cached_freqs.shape[0] >= 8)
        tt.assert_close(actual, expected, msg=f'expected {expected}, but got {actual}')
        self.assertEqual(expected.size(), actual.size())

        actual = self.rotary_embedding(seqs, seq_len=8, offset=4)

        self.assertTensorsEqual(expected, actual)

        actual = self.rotary_embedding(seqs, seq_len=6, offset=2)

        self.assertNotEqual(expected.size(), actual.size())

        with self.assertRaises(AssertionError):
            tt.assert_close(actual, expected, msg=f'expected {expected}, but got {actual}')

        self.rotary_embedding.tmp_store('cached_freqs', None) # clear the cache

    def test_pixel_rotary_embedding_forward_doesnt_cache(self):
        q_or_k = self.queries_1_4_8.repeat(1, 2, 1) # shape (1, 8, 8)

        seqs = self.pixel_rotary_embedding.get_seq_pos(q_or_k.size(1), q_or_k.device, q_or_k.dtype)

        self.assertTrue(self.pixel_rotary_embedding.cached_freqs is None or self.pixel_rotary_embedding.cached_freqs.shape[0] == 0)

        # seq_len specified would normally cause caching
        self.pixel_rotary_embedding(seqs, seq_len=8)

        self.assertTrue(self.pixel_rotary_embedding.cached_freqs is None or self.pixel_rotary_embedding.cached_freqs.shape[0] == 0)

    def test_learned_freq_rotary_embedding_forward_doesnt_cache(self):
        q_or_k = self.queries_1_4_8.repeat(1, 2, 1) # shape (1, 8, 8)

        seqs = self.learned_freq_rotary_embedding.get_seq_pos(q_or_k.size(1), q_or_k.device, q_or_k.dtype)

        self.assertTrue(self.learned_freq_rotary_embedding.cached_freqs is None or self.learned_freq_rotary_embedding.cached_freqs.shape[0] == 0)

        # seq_len specified would normally cause caching
        self.learned_freq_rotary_embedding(seqs, seq_len=8)

        self.assertTrue(self.learned_freq_rotary_embedding.cached_freqs is None or self.learned_freq_rotary_embedding.cached_freqs.shape[0] == 0)

    def test_rotary_embedding_init_freqs_calculation(self):
        # default freqs_for is 'lang'
        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim)

        actual = rotary_embedding.freqs
        expected = torch.tensor([1.0000000000000000, 0.1000000014901161, 0.0099999997764826, 0.0010000000474975])

        self.assertTensorsEqual(expected, actual)

        # custom freqs overrides any calculations for freqs
        expected = torch.tensor([1., 2., 3., 4.])
        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim, custom_freqs=expected)

        actual = rotary_embedding.freqs

        self.assertTensorsEqual(expected, actual)

        # this allows for normally invalid data states
        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim // 2, custom_freqs=expected)

        actual = rotary_embedding.freqs

        self.assertTensorsEqual(expected, actual)

        # freqs_for = 'pixel'
        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim, freqs_for='pixel')

        actual = rotary_embedding.freqs
        expected = torch.tensor([3.1415927410125732, 7.3303837776184082, 11.5191726684570312, 15.7079639434814453])

        # max_freq is used for 'pixel' only; double the default for this case
        # notably not an actual upper bound for the values in the calculated freqs tensor
        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim, freqs_for='pixel', max_freq=20)

        actual = rotary_embedding.freqs
        expected = torch.tensor([3.1415927410125732, 12.5663709640502930, 21.9911499023437500, 31.4159278869628906])

        self.assertTensorsEqual(expected, actual)

        self.assertTensorsEqual(expected, actual)

        # freqs_for = 'constant'
        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim, freqs_for='constant')

        actual = rotary_embedding.freqs
        expected = torch.tensor([1.])

        self.assertTensorsEqual(expected, actual)

        # num_freqs only used for constant freq tensor creation
        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim, freqs_for='constant', num_freqs=3)

        actual = rotary_embedding.freqs
        expected = torch.tensor([1., 1., 1.])

        self.assertTensorsEqual(expected, actual)

        # specifying theta; affects calculation of freqs for 'lang' only
        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim, theta=10000)

        actual = rotary_embedding.freqs
        expected = torch.tensor([1.0000000000000000, 0.1000000014901161, 0.0099999997764826, 0.0010000000474975])

        self.assertTensorsEqual(expected, actual)

        # double theta to see effects
        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim, theta=20000)

        actual = rotary_embedding.freqs
        expected = torch.tensor([1.0000000000000000e+00, 8.4089644253253937e-02, 7.0710680447518826e-03, 5.9460353804752231e-04])

        self.assertTensorsEqual(expected, actual)

        # half theta to see effects
        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim, theta=5000)

        actual = rotary_embedding.freqs
        expected = torch.tensor([1.0000000000000000, 0.1189207136631012, 0.0141421360895038, 0.0016817927826196])

        self.assertTensorsEqual(expected, actual)

        # theta_rescale_factor affects calculation of freqs for 'lang' only
        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim, theta=10000, theta_rescale_factor=0.5)

        actual = rotary_embedding.freqs
        expected = torch.tensor([1.0000000000000000, 0.1259921044111252, 0.0158740114420652, 0.0020000000949949])

        self.assertTensorsEqual(expected, actual)

        # double dim
        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim * 2, theta=10000)

        actual = rotary_embedding.freqs
        expected = torch.tensor([1.0000000000000000e+00, 3.1622776389122009e-01, 1.0000000149011612e-01,
        3.1622778624296188e-02, 9.9999997764825821e-03, 3.1622778624296188e-03,
        1.0000000474974513e-03, 3.1622778624296188e-04])

        self.assertTensorsEqual(expected, actual)

    def test_rotary_embedding_init_default_seq_dim(self):
        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim)

        self.assertEqual(rotary_embedding.default_seq_dim, -2)

        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim, seq_before_head_dim=True)

        self.assertEqual(rotary_embedding.default_seq_dim, -3)

    def test_rotary_embedding_init_throws_invalid_interpolate_factor(self):
        with self.assertRaises(AssertionError):
            rotary_embedding_torch.RotaryEmbedding(self.dim, interpolate_factor=0)

        with self.assertRaises(AssertionError):
            rotary_embedding_torch.RotaryEmbedding(self.dim, interpolate_factor=-1)

    def test_rotary_embedding_init_not_xpos_add_none_scale_attr(self):
        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim)

        self.assertTrue(hasattr(rotary_embedding, 'scale'))
        self.assertIsNone(rotary_embedding.scale)

    def test_rotary_embedding_init_xpos(self):
        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim, use_xpos=True)

        actual = rotary_embedding.scale
        expected = torch.tensor([0.2857142984867096, 0.4642857015132904, 0.6428571343421936, 0.8214285969734192])

        self.assertTensorsEqual(expected, actual)

        rotary_embedding = rotary_embedding_torch.RotaryEmbedding(self.dim * 2, use_xpos=True)

        actual = rotary_embedding.scale
        expected = torch.tensor([0.2857142984867096, 0.3750000000000000, 0.4642857015132904, 0.5535714030265808, 
        0.6428571343421936, 0.7321428656578064, 0.8214285969734192, 0.9107142686843872])

        self.assertTensorsEqual(expected, actual)
