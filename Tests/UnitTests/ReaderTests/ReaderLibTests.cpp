//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"

#include "BlockRandomizer.h"
#include "DataDeserializer.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(ReaderLibTests)

class MockDeserializer : public DataDeserializer
{
private:
    SequenceDescriptions m_sequenceDescriptions;

public:
    std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override
    {
        std::vector<StreamDescriptionPtr> result;
        return result;
    }

    const SequenceDescriptions& GetSequenceDescriptions() const override
    {
        return m_sequenceDescriptions;
    }

    void StartEpoch(const EpochConfiguration& config) override
    {
        UNREFERENCED_PARAMETER(config);
    }

    std::vector<std::vector<SequenceDataPtr>> GetSequencesById(const std::vector<size_t>& ids) override
    {
        UNREFERENCED_PARAMETER(ids);
        return std::vector<std::vector<SequenceDataPtr>>();
    }

    void RequireChunk(size_t chunkIndex) override
    {
        UNREFERENCED_PARAMETER(chunkIndex);
    }

    void ReleaseChunk(size_t chunkIndex) override
    {
        UNREFERENCED_PARAMETER(chunkIndex);
    }
};

BOOST_AUTO_TEST_CASE(BlockRandomizerInstantiate)
{
    auto mockDeserializer = std::make_shared<MockDeserializer>();

    auto randomizer = std::make_shared<BlockRandomizer>(0, SIZE_MAX, mockDeserializer);
}

class NoRandomizer
{
public:
    NoRandomizer(const SequenceDescriptions& timeline)
        : m_timeline(timeline),
          m_numSamples(0),
          m_numSequences(timeline.size()) // TODO assumes all valid
    {

        for (auto & seqDesc : m_timeline)
        {
            m_numSamples += seqDesc->m_numberOfSamples;
        }
    }

    ~NoRandomizer()
    {

    }

    NoRandomizer(const NoRandomizer&) = delete;
    NoRandomizer& operator=(const NoRandomizer&) = delete;

    size_t GetNumSequences() const
    {
        return m_numSequences;
    }

    size_t GetNumSamples() const
    {
        return m_numSamples;
    }

    size_t GetSamplePosition() const
    {
        return m_samplePosition;
    }

    size_t GetSequencePosition() const
    {
        return m_sequencePosition;
    }

    void SetSamplePosition(size_t globalSamplePosition)
    {
        size_t requestedSamplePosition = globalSamplePosition % m_numSamples;

        size_t newSequencePosition = 0;
        size_t newSamplePosition = 0;

        // Find the next sequence start after the requested sample position
        // (potentially wrapping around)
        while ((newSamplePosition < requestedSamplePosition) &&
               (newSequencePosition < m_numSequences))
        {
            newSamplePosition += m_timeline[newSequencePosition]->m_numberOfSamples;
            newSequencePosition++;
        }

        if (newSequencePosition == m_numSequences)
        {
            newSamplePosition = 0;
            newSequencePosition = 0;
        }

        m_samplePosition = newSamplePosition;
        m_sequencePosition = newSequencePosition;
    }

    SequenceDescriptions GetNextSequenceDescriptions(size_t maxSampleCount, bool dropPartial)
    {
        SequenceDescriptions result;

        size_t currentSampleCount = 0;

        do
        {
            auto& seqDesc = m_timeline[m_sequencePosition];
            // We need to take at least one sequence, even if exceeding.
            // Otherwise, take an additional one if it fits.
            bool takeOne = currentSampleCount == 0 ||
                (currentSampleCount + seqDesc->m_numberOfSamples <= maxSampleCount);

            if (takeOne)
            {
                result.push_back(seqDesc);
            }

            currentSampleCount += seqDesc->m_numberOfSamples;

            if (takeOne || dropPartial)
            {
                m_sequencePosition++;
                m_samplePosition += seqDesc->m_numberOfSamples;
                if (m_sequencePosition == m_numSequences)
                {
                    m_sequencePosition = 0;
                    m_samplePosition = 0;
                }
            }
        } while (currentSampleCount < maxSampleCount);

        return std::move(result);
    }

private:
    const SequenceDescriptions& m_timeline;
    size_t m_numSamples;
    size_t m_numSequences;
    size_t m_sequencePosition;
    size_t m_samplePosition;
};

BOOST_AUTO_TEST_CASE(NoRandomizerWip)
{
    SequenceDescriptions timeline;
    std::array<SequenceDescription, 4> sequenceDescriptions = {
        SequenceDescription{ 0, 1, 0, true },
        SequenceDescription{ 1, 1, 0, true },
        SequenceDescription{ 2, 2, 0, true },
        SequenceDescription{ 3, 2, 0, true }
    };
    timeline.push_back(&sequenceDescriptions[0]);
    timeline.push_back(&sequenceDescriptions[1]);
    timeline.push_back(&sequenceDescriptions[2]);
    timeline.push_back(&sequenceDescriptions[3]);

    auto noRandomizer = std::make_shared<NoRandomizer>(timeline);
    BOOST_CHECK(noRandomizer->GetNumSequences() == 4);
    BOOST_CHECK(noRandomizer->GetNumSamples() == 6);

    noRandomizer->SetSamplePosition(0);
    BOOST_CHECK(noRandomizer->GetSamplePosition() == 0);
    BOOST_CHECK(noRandomizer->GetSequencePosition() == 0);

    noRandomizer->SetSamplePosition(3);
    BOOST_CHECK(noRandomizer->GetSamplePosition() == 4);
    BOOST_CHECK(noRandomizer->GetSequencePosition() == 3);

    noRandomizer->SetSamplePosition(5);
    BOOST_CHECK(noRandomizer->GetSamplePosition() == 0);
    BOOST_CHECK(noRandomizer->GetSequencePosition() == 0);

    SequenceDescriptions result;
    noRandomizer->SetSamplePosition(0);
    result = noRandomizer->GetNextSequenceDescriptions(0, true);
    BOOST_CHECK(result.size() == 1);
    BOOST_CHECK(result[0]->m_id == 0);
    BOOST_CHECK(noRandomizer->GetSamplePosition() == 1);
    BOOST_CHECK(noRandomizer->GetSequencePosition() == 1);

    noRandomizer->SetSamplePosition(0);
    result = noRandomizer->GetNextSequenceDescriptions(1, true);
    BOOST_CHECK(result.size() == 1);
    BOOST_CHECK(result[0]->m_id == 0);
    BOOST_CHECK(noRandomizer->GetSamplePosition() == 1);
    BOOST_CHECK(noRandomizer->GetSequencePosition() == 1);

    noRandomizer->SetSamplePosition(0);
    result = noRandomizer->GetNextSequenceDescriptions(2, true);
    BOOST_CHECK(result.size() == 2);
    BOOST_CHECK(result[0]->m_id == 0);
    BOOST_CHECK(result[1]->m_id == 1);
    BOOST_CHECK(noRandomizer->GetSamplePosition() == 2);
    BOOST_CHECK(noRandomizer->GetSequencePosition() == 2);

    noRandomizer->SetSamplePosition(4);
    result = noRandomizer->GetNextSequenceDescriptions(3, true);
    BOOST_CHECK(result.size() == 2);
    BOOST_CHECK(result[0]->m_id == 3);
    BOOST_CHECK(result[1]->m_id == 0);
    BOOST_CHECK(noRandomizer->GetSamplePosition() == 1);
    BOOST_CHECK(noRandomizer->GetSequencePosition() == 1);

    noRandomizer->SetSamplePosition(2);
    result = noRandomizer->GetNextSequenceDescriptions(3, false);
    BOOST_CHECK(result.size() == 1);
    BOOST_CHECK(result[0]->m_id == 2);
    BOOST_CHECK(noRandomizer->GetSamplePosition() == 4);
    BOOST_CHECK(noRandomizer->GetSequencePosition() == 3);
}

BOOST_AUTO_TEST_SUITE_END()

} } } }
