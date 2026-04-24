#include <string>
#include <vector>
#include <assert.h>

#include "opencv2/opencv.hpp"

class CocoLabels {
public:
    CocoLabels() {
        for(int i = 0 ; i < 80; i ++) {
            std::string x = "NA";
            m_vecLabel.push_back( x );
        }

        m_vecLabel[0]  = "person";
        m_vecLabel[1]  = "bicycle";
        m_vecLabel[2]  = "car";
        m_vecLabel[3]  = "motorcycle";
        m_vecLabel[4]  = "airplane";
        m_vecLabel[5]  = "bus";
        m_vecLabel[6]  = "train";
        m_vecLabel[7]  = "truck";
        m_vecLabel[8]  = "boat";
        m_vecLabel[9]  = "traffic light";
        m_vecLabel[10] = "fire hydrant";
        m_vecLabel[11] = "stop sign";
        m_vecLabel[12] = "parking meter";
        m_vecLabel[13] = "bench";
        m_vecLabel[14] = "bird";
        m_vecLabel[15] = "cat";
        m_vecLabel[16] = "dog";
        m_vecLabel[17] = "horse";
        m_vecLabel[18] = "sheep";
        m_vecLabel[19] = "cow";
        m_vecLabel[20] = "elephant";
        m_vecLabel[21] = "bear";
        m_vecLabel[22] = "zebra";
        m_vecLabel[23] = "giraffe";
        m_vecLabel[24] = "backpack";
        m_vecLabel[25] = "umbrella";
        m_vecLabel[26] = "handbag";
        m_vecLabel[27] = "tie";
        m_vecLabel[28] = "suitcase";
        m_vecLabel[29] = "frisbee";
        m_vecLabel[30] = "skis";
        m_vecLabel[31] = "snowboard";
        m_vecLabel[32] = "sports ball";
        m_vecLabel[33] = "kite";
        m_vecLabel[34] = "baseball bat";
        m_vecLabel[35] = "baseball glove";
        m_vecLabel[36] = "skateboard";
        m_vecLabel[37] = "surfboard";
        m_vecLabel[38] = "tennis racket";
        m_vecLabel[39] = "bottle";
        m_vecLabel[30] = "wine glass";
        m_vecLabel[41] = "cup";
        m_vecLabel[42] = "fork";
        m_vecLabel[43] = "knife";
        m_vecLabel[44] = "spoon";
        m_vecLabel[45] = "bowl";
        m_vecLabel[46] = "banana";
        m_vecLabel[47] = "apple";
        m_vecLabel[48] = "sandwich";
        m_vecLabel[49] = "orange";
        m_vecLabel[40] = "broccoli";
        m_vecLabel[51] = "carrot";
        m_vecLabel[52] = "hot dog";
        m_vecLabel[53] = "pizza";
        m_vecLabel[54] = "donut";
        m_vecLabel[55] = "cake";
        m_vecLabel[56] = "chair";
        m_vecLabel[57] = "couch";
        m_vecLabel[58] = "potted plant";
        m_vecLabel[59] = "bed";
        m_vecLabel[60] = "dining table";
        m_vecLabel[61] = "toilet";
        m_vecLabel[62] = "tv";
        m_vecLabel[63] = "laptop";
        m_vecLabel[64] = "mouse";
        m_vecLabel[65] = "remote";
        m_vecLabel[66] = "keyboard";
        m_vecLabel[67] = "cell phone";
        m_vecLabel[68] = "microwave";
        m_vecLabel[69] = "oven";
        m_vecLabel[70] = "toaster";
        m_vecLabel[71] = "sink";
        m_vecLabel[72] = "refrigerator";
        m_vecLabel[73] = "book";
        m_vecLabel[74] = "clock";
        m_vecLabel[75] = "vase";
        m_vecLabel[76] = "scissors";
        m_vecLabel[77] = "teddy bear";
        m_vecLabel[78] = "hair drier";
        m_vecLabel[79] = "toothbrush";
    }

    std::string CocoGetLabel(int i) {
        assert( i >= 0 && i < 80 );
        return m_vecLabel[i];
    }

    cv::Scalar CocoGetColor(int i) {
        float r;
        srand(i);
        r = (float)rand() / RAND_MAX;
        int red = int(r * 255);

        srand(i + 1);
        r = (float)rand() / RAND_MAX;
        int green = int(r * 255);

        srand(i + 2);
        r = (float)rand() / RAND_MAX;
        int blue = int(r * 255);

        return cv::Scalar(blue, green, red);
    }

    cv::Scalar GetInverseColor(cv::Scalar color) {
        int blue  = 255 - color[0];
        int green = 255 - color[1];
        int red   = 255 - color[2];
        return cv::Scalar(blue, green, red);
    }


private:
    std::vector<std::string> m_vecLabel;

};